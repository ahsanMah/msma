import csv
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_train_test_data
from losses.losses import dsm_loss, ocnn_loss, update_radius, normalized_dsm_loss

SIGMA_LEVELS = None

@tf.function
def test_step_masked(model, data_batch, sigma_levels, num_L):
    idx_sigmas = (num_L-1)*tf.ones([data_batch.shape[0]],
                                        dtype=tf.dtypes.int32)
    sigmas = tf.gather(sigma_levels, idx_sigmas)
    sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))

    # Perturb and apply mask
    x_batch, masks = tf.split(data_batch, 2, axis=-1)
    x_perturbed = x_batch + tf.random.normal(shape=x_batch.shape) * sigmas
    x_perturbed = tf.multiply(x_perturbed, masks)
    data_batch_perturbed = tf.concat((x_perturbed, masks), axis=-1)

    scores = model([data_batch_perturbed, idx_sigmas])
    # current_loss = normalized_dsm_loss(scores, x_perturbed, x_batch, sigmas, masks)
    current_loss = dsm_loss(scores, x_perturbed, x_batch, sigmas)
    return current_loss


@tf.function
def train_ocnn_step(model, optimizer, data_batch_perturbed, idx_sigmas, r):
    with tf.GradientTape() as t:
        w = model.get_layer(name="embedding").weights
        V = model.get_layer(name="distance").weights
        scores = model([data_batch_perturbed, idx_sigmas])
        current_loss = ocnn_loss(r, w, V, scores)
        gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    r = update_radius(scores)
    return current_loss, r

def main():

    device = utils.get_tensorflow_device()
    tf.random.set_seed(2019)
    LOG_FREQ = 100
    SIGMA_LEVELS = utils.get_sigma_levels()
    NUM_L = configs.config_values.num_L
    
    if configs.config_values.y_cond or configs.config_values.model == "masked_refinenet":
        SPLITS = utils.dict_splits[configs.config_values.dataset]

    @tf.function
    def test_one_step(model, data_batch):
        idx_sigmas = (NUM_L-1) * tf.ones([data_batch.shape[0]],
                                        dtype=tf.dtypes.int32)
        sigmas = tf.gather(SIGMA_LEVELS, idx_sigmas)
        sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
        data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas
        scores = model([data_batch_perturbed, idx_sigmas])
        current_loss = dsm_loss(scores, data_batch_perturbed, data_batch, sigmas)
        return current_loss

    @tf.function
    def train_one_step(model, optimizer, data_batch):
        idx_sigmas = tf.random.uniform([data_batch.shape[0]], minval=0,
                                        maxval=NUM_L,
                                        dtype=tf.dtypes.int32)
        sigmas = tf.gather(SIGMA_LEVELS, idx_sigmas)
        sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
        data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas
        
        with tf.GradientTape() as t:
            scores = model([data_batch_perturbed, idx_sigmas])
            current_loss = dsm_loss(scores, data_batch_perturbed, data_batch, sigmas)
            gradients = t.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    @tf.function
    def test_step_masked(model, data_batch):
        idx_sigmas = (NUM_L-1)*tf.ones([data_batch.shape[0]],
                                            dtype=tf.dtypes.int32)
        sigmas = tf.gather(SIGMA_LEVELS, idx_sigmas)
        sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))

        # Perturb and apply mask
        x_batch, masks = tf.split(data_batch, SPLITS, axis=-1)
        perturbation =  tf.random.normal(shape=x_batch.shape) * sigmas
        # perturbation = tf.multiply(perturbation, masks)
        x_perturbed = x_batch + perturbation
        data_batch_perturbed = tf.concat((x_perturbed, masks), axis=-1)

        scores = model([data_batch_perturbed, idx_sigmas])
        current_loss = dsm_loss(scores, x_perturbed, x_batch, sigmas)
        return current_loss

    @tf.function
    def train_one_masked_step(model, optimizer, data_batch):
        idx_sigmas = tf.random.uniform([data_batch.shape[0]], minval=0,
                                    maxval=NUM_L,
                                    dtype=tf.dtypes.int32)
        sigmas = tf.gather(SIGMA_LEVELS, idx_sigmas)

        # --> Noise is only applied to foreground
        # Conditioned Noise..?
        x_batch, masks = tf.split(data_batch, SPLITS, axis=-1)
        sigmas = tf.reshape(sigmas, shape=(x_batch.shape[0], 1, 1, 1))
        perturbation =  tf.random.normal(shape=x_batch.shape) * sigmas
        # perturbation = tf.multiply(perturbation, masks)
        x_perturbed = x_batch + perturbation
        data_batch_perturbed = tf.concat((x_perturbed, masks), axis=-1)

        with tf.GradientTape() as t:
            scores = model([data_batch_perturbed, idx_sigmas])
            # current_loss = dsm_loss(scores, x_perturbed, x_batch, sigmas, masks)
            current_loss = dsm_loss(scores, x_perturbed, x_batch, sigmas)
            gradients = t.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return current_loss

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    basename = "logs/{model}/{dataset}/{time}".format(
        model=configs.config_values.model,
        dataset=configs.config_values.dataset,
        time=start_time
    )
    train_log_dir = basename + '/train'
    test_log_dir  = basename + '/test'

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    
    # load dataset from tfds (or use downloaded version if exists)
    train_data, test_data = get_train_test_data(configs.config_values.dataset)
    # train_data = train_data.cache()

    # # split data into batches
    train_data = train_data.shuffle(buffer_size=10000)
    if configs.config_values.dataset != 'celeb_a':
        train_data = train_data.batch(configs.config_values.batch_size)
    train_data = train_data.repeat()
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_data = test_data.shuffle(buffer_size=10000)
    test_data = test_data.batch(configs.config_values.batch_size*4)
    test_data = test_data.take(4).cache()

    # Just a larger batch size of the same dataset
    ocnn_batch_size = configs.config_values.batch_size*4
    if configs.config_values.ocnn:    
        ocnn_data =  get_train_test_data(configs.config_values.dataset)[0]
        # ocnn_data = ocnn_data.cache()
        ocnn_data = ocnn_data.shuffle(buffer_size=10000)
        ocnn_data = ocnn_data.batch(ocnn_batch_size)
        ocnn_data = ocnn_data.repeat()
        ocnn_data = ocnn_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()
    # save_dir += "/multichannel/"

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    sigma_levels = utils.get_sigma_levels()

    model, optimizer, step, ocnn_model, ocnn_optimizer = utils.try_load_model(save_dir,
     step_ckpt=configs.config_values.resume_from, verbose=True, ocnn=configs.config_values.ocnn)

    # Save checkpoint
    ckpt = None
    if configs.config_values.ocnn:
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model,
                                ocnn_model=ocnn_model, ocnn_optmizer=ocnn_optimizer)
    else:
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)

    manager = tf.train.CheckpointManager(ckpt, directory=save_dir,
        max_to_keep=configs.config_values.max_to_keep)
    
    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step + 1)
    progress_bar.set_description('current loss ?')

    steps_per_epoch = utils.dict_train_size[configs.config_values.dataset] // configs.config_values.batch_size
    ocnn_freq = 25 * steps_per_epoch # Every 25 epochs 
    ocnn_steps_per_epoch = utils.dict_train_size[configs.config_values.dataset] // ocnn_batch_size

    radius = 1.0
    loss_history = []
    epoch =  step // steps_per_epoch

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    with tf.device(device):  # For some reason, this makes everything faster
        avg_loss = 0
        for data_batch in progress_bar:

            if step % steps_per_epoch == 0:
                epoch += 1

            step += 1
       
            train_step = None
            if configs.config_values.y_cond or configs.config_values.model == "masked_refinenet":
                train_step, test_step = train_one_masked_step, test_step_masked
            else:
                train_step, test_step = train_one_step, test_one_step

            current_loss = train_step(model, optimizer, data_batch)
            train_loss(current_loss)
            
            progress_bar.set_description('[epoch {:d}] | current loss {:.3f}'.format(
                epoch, train_loss.result().numpy()
            ))

            if step % LOG_FREQ == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=step)
                
                for x_test in test_data:
                    _loss = test_step(model, data_batch)
                    test_loss(_loss)
                
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=step)
                
                # Reset metrics every epoch
                train_loss.reset_states()
                test_loss.reset_states()
            
            # loss_history.append([step, current_loss.numpy()])
            avg_loss += current_loss

            if configs.config_values.ocnn and step % ocnn_freq == 0:
                for i in range(ocnn_steps_per_epoch):
                    data_batch = next(iter(ocnn_data))
                    best_idx_sigmas = tf.ones([data_batch.shape[0]],
                    dtype=tf.dtypes.int32) * configs.config_values.num_L-1
                    sigmas = tf.gather(sigma_levels, best_idx_sigmas)
                    sigmas = tf.reshape(sigmas, shape=(data_batch.shape[0], 1, 1, 1))
                    data_batch_perturbed = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas
                    
                    loss, radius = train_ocnn_step(ocnn_model, ocnn_optimizer, data_batch_perturbed, best_idx_sigmas, radius)
                    progress_bar.set_description(
                        'OC-NN: radius {:.3f} | loss {:.3f}'.format(radius, loss))

            if step % configs.config_values.checkpoint_freq == 0:

                ckpt.step.assign(step)
                manager.save()

                # Append in csv file
                # with open(save_dir + 'loss_history.csv', mode='a', newline='') as csv_file:
                #     writer = csv.writer(csv_file, delimiter=';')
                #     writer.writerows(loss_history)

                print(
                    "\nSaved checkpoint. Average loss: {:.3f}".format(avg_loss / configs.config_values.checkpoint_freq))
                loss_history = []
                avg_loss = 0

            if step == total_steps:
                return
