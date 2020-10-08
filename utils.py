import argparse
import os
import re

import tensorflow as tf

import configs
from model.refinenet import RefineNet, RefineNetTwoResidual, MaskedRefineNet
from model.resnet import ResNet

dict_datasets_image_size = {
    "circles": (64,64,1),
    "highres": (2048,1024,3),
    "brain": (91,109,1),
    "masked_brain": (91,109,2),
    "seg_brain": (91,109,8),
    "pet" : (64,64,3),
    "masked_pet" : (64,64,4),
    "blown_fashion": (56, 56, 1),
    "blown_masked_fashion": (56, 56, 2),
    'masked_fashion': (28, 28, 2),
    'fashion_mnist': (28, 28, 1),
    'mnist_ood': (28, 28, 1),
    'mnist': (28, 28, 1),
    'cifar10': (32, 32, 3),
    "masked_cifar10": (32,32,4),
    "seg_cifar10": (32,32,14),
    "multiscale_cifar10": (32,32,6),
    'celeb_a': (32, 32, 3),
    "svhn_cropped": (32, 32, 3),
}

dict_train_size = {
    "circles": 100000,
    "svhn_cropped": 73000,
    'cifar10': 60000,
    "brain": 10500,
    "masked_brain": 10500,
    "seg_brain": 10500,
    "masked_cifar10": 40000,
    "seg_cifar10": 40000,
    "multiscale_cifar10":50000,
    "pet" : 6500,
    "masked_pet" : 6500,
    "blown_fashion": 60000,
    "blown_masked_fashion": 60000,
    'masked_fashion': 60000,
    'fashion_mnist': 60000,
    'mnist_ood': 60000,
    'mnist': 60000,
}

dict_splits = {
    "masked_fashion": (1,1),
    "masked_brain": (1,1),
    "seg_brain": (1,7),
    "masked_cifar10": (3,1),
    "seg_cifar10": (3,11),
    "multiscale_cifar10": (3,3)
}


def find_k_closest(image, k, data_as_array):
    l2_distances = tf.reduce_sum(tf.square(data_as_array - image), axis=[1, 2, 3])
    _, smallest_idx = tf.math.top_k(-l2_distances, k)
    closest_k = tf.gather(data_as_array, smallest_idx[:k])
    return closest_k, smallest_idx[:k]


def get_dataset_image_size(dataset_name):
    return dict_datasets_image_size[dataset_name]


def check_args_validity(args):
    assert args.model in ["baseline", "resnet", "refinenet", "refinenet_twores", "masked_refinenet"]
    if args.max_to_keep == -1:
        args.max_to_keep = None
    args.split = args.split.split(",")
    args.split = list(map(lambda x: x.strip(), args.split))
    return

def _build_parser():
    parser = argparse.ArgumentParser(description='I AM A HELP MESSAGE')
    parser.add_argument('--experiment', default='train', help="what experiment to run (default: train)")
    parser.add_argument('--dataset', default='mnist',
                        help="tfds name of dataset (default: 'mnist')")
    parser.add_argument('--model', default='refinenet',
                        help="Model to use. Can be \'refinenet\', \'resnet\', \'baseline\' (default: refinenet)")
    parser.add_argument('--filters', default=128, type=int,
                        help='number of filters in the model. (default: 128)')
    parser.add_argument('--num_L', default=10, type=int,
                        help="number of levels of noise to use (default: 10)")
    parser.add_argument('--sigma_low', default=0.01, type=float,
                        help="lowest value for noise (default: 0.01)")
    parser.add_argument('--sigma_high', default=1.0, type=float,
                        help="highest value for noise (default: 1.0)")
    parser.add_argument('--sigma_sequence', default="geometric", type=str,
                        help="can be \'geometric\' or \'linear\' (default: geometric)")
    parser.add_argument('--steps', default=200000, type=int,
                        help="number of steps to train the model for (default: 200000)")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="learning rate for the optimizer")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="batch size (default: 128)")
    parser.add_argument('--samples_dir', default='./samples/',
                        help="folder for saving samples (default: ./samples/)")
    parser.add_argument('--checkpoint_dir', default='./saved_models/',
                        help="folder for saving model checkpoints (default: ./saved_models/)")
    parser.add_argument('--checkpoint_freq', default=5000, type=int,
                        help="how often to save a model checkpoint (default: 5000 iterations)")
    parser.add_argument('--resume', action='store_false',
                        help="whether to resume from latest checkpoint (default: True)")
    parser.add_argument('--resume_from', default=-1, type=int,
                        help='Step of checkpoint where to resume the model from. (default: latest one)')
    parser.add_argument('--init_samples', default="",
                        help="Folder with images to be used as x0 for sampling with annealed langevin dynamics")
    parser.add_argument('--k', default=10, type=int,
                        help='number of nearest neighbours to find from data (default: 10)')
    parser.add_argument('--eval_setting', default="sample", type=str,
                        help="can be \'sample\' or \'fid\' (default: sample)")
    parser.add_argument('--ocnn', action='store_true',
                        help="whether to attach an ocnn to the model (default: False)")
    parser.add_argument('--y_cond', action='store_true',
                        help="whether the model is conditioned on auxiallary y information (default: False)")
    parser.add_argument('--max_to_keep', default=2, type=int,
                        help="Number of checkopints to keep saved (default: 2)")
    parser.add_argument('--split', default='100,0', type=str,
                        help="Train/(Tune)/Test split e.g. 'train[:90%],train[-10%:],test' (default: train,test)")

    return parser

def get_command_line_args():
    parser = _build_parser()

    parser = parser.parse_args()

    check_args_validity(parser)

    print("=" * 20 + "\nParameters: \n")
    for key in parser.__dict__:
        print(key + ': ' + str(parser.__dict__[key]))
    print("=" * 20 + "\n")
    return parser


def get_tensorflow_device():
    device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
    print("Using device {}".format(device))
    return device


def get_savemodel_dir():
    models_dir = configs.config_values.checkpoint_dir
    model_name = configs.config_values.model
    
    # Folder name: model_name+filters+dataset+L
    if not configs.config_values.model == 'baseline':
        complete_model_name = '{}{}_{}_L{}_SH{:.0e}_SL{:.0e}/train_{}'.format(model_name, configs.config_values.filters,
                                                   configs.config_values.dataset, configs.config_values.num_L,
                                                   configs.config_values.sigma_high,
                                                   configs.config_values.sigma_low,
                                                   "_".join(configs.config_values.split)
                                                   )
    else:
        complete_model_name = '{}{}_{}_SL{:.0e}'.format(model_name, configs.config_values.filters, configs.config_values.dataset,configs.config_values.sigma_low)
    folder_name = models_dir + complete_model_name + '/'
    
    if configs.config_values.ocnn:
        folder_name += "ocnn/"
    
    return folder_name, complete_model_name


def evaluate_print_model_summary(model, verbose=True):
    batch = 1
    input_shape = (batch,) + get_dataset_image_size(configs.config_values.dataset)
    print(input_shape)
    x = [tf.ones(shape=input_shape), tf.ones(batch, dtype=tf.int32)]
    model(x)
    if verbose:
        print(model.summary())

def attach_ocnn(top=True, encoding=False):
    pass


def try_load_model(save_dir, step_ckpt=-1, return_new_model=True, verbose=True, ocnn=False):
    """
    Tries to load a model from the provided directory, otherwise returns a new initialized model.
    :param save_dir: directory with checkpoints
    :param step_ckpt: step of checkpoint where to resume the model from
    :param verbose: true for printing the model summary
    :return:
    """
    ocnn_model=None
    ocnn_optimizer=None

    import tensorflow as tf
    tf.compat.v1.enable_v2_behavior()
    if configs.config_values.model == 'baseline':
        configs.config_values.num_L = 1

    splits=False
    if configs.config_values.y_cond:
        splits = dict_splits[configs.config_values.dataset]

    # initialize return values
    model_name = configs.config_values.model
    if model_name == 'resnet':
        model = ResNet(filters=configs.config_values.filters, activation=tf.nn.elu)
    elif model_name in ['refinenet', 'baseline']:
        model = RefineNet(filters=configs.config_values.filters, activation=tf.nn.elu,
        y_conditioned=configs.config_values.y_cond, splits=splits)
    elif model_name == 'refinenet_twores':
        model = RefineNetTwoResidual(filters=configs.config_values.filters, activation=tf.nn.elu)
    elif model_name == 'masked_refinenet':
        print("Using Masked RefineNet...")
        # assert configs.config_values.y_cond 
        model = MaskedRefineNet(filters=configs.config_values.filters, activation=tf.nn.elu, 
        splits=dict_splits[configs.config_values.dataset], y_conditioned=configs.config_values.y_cond)

    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.config_values.learning_rate)
    step = 0
    evaluate_print_model_summary(model, verbose)
    
    if ocnn:
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Flatten, Dense, AvgPool2D
        # Building OCNN on top
        print("Building OCNN...")
        Input = [Input(name="images", shape=(28,28,1)),
                Input(name="idx_sigmas", shape=(), dtype=tf.int32)]

        score_logits = model(Input)
        x = Flatten()(score_logits)
        x = Dense(128, activation="linear", name="embedding")(x)
        dist = Dense(1, activation="linear", name="distance")(x)
        ocnn_model = Model(inputs=Input, outputs=dist, name="OC-NN")
        ocnn_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        evaluate_print_model_summary(ocnn_model, verbose=True)

    # if resuming training, overwrite model parameters from checkpoint
    if configs.config_values.resume:
        if step_ckpt == -1:
            print("Trying to load latest model from " + save_dir)
            checkpoint = tf.train.latest_checkpoint(str(save_dir))
        else:
            print("Trying to load checkpoint with step", step_ckpt, " model from " + save_dir)
            onlyfiles = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            # r = re.compile(".*step_{}-.*".format(step_ckpt))
            r = re.compile("ckpt-{}\\..*".format(step_ckpt))

            name_all_checkpoints = sorted(list(filter(r.match, onlyfiles)))
            print(name_all_checkpoints)
            # Retrieve name of the last checkpoint with that number of steps
            name_ckpt = name_all_checkpoints[-1][:-6]
            # print(name_ckpt)
            checkpoint = save_dir + name_ckpt
        if checkpoint is None:
            print("No model found.")
            if return_new_model:
                print("Using a new model")
            else:
                print("Returning None")
                model = None
                optimizer = None
                step = None
        else:
            step = tf.Variable(0)

            if ocnn:
                ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model,
                ocnn_model=ocnn_model, ocnn_optimizer=ocnn_optimizer)
            else:
                 ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)

            ckpt.restore(checkpoint)
            step = int(step)
            print("Loaded model: " + checkpoint)

    return model, optimizer, step, ocnn_model, ocnn_optimizer


def get_sigma_levels():
    if configs.config_values.model == 'baseline':
        sigma_levels = tf.ones(1) * configs.config_values.sigma_low
    elif configs.config_values.sigma_sequence == 'linear':
        sigma_levels = tf.linspace(configs.config_values.sigma_high,
                                   configs.config_values.sigma_low,
                                   configs.config_values.num_L)
    elif configs.config_values.sigma_sequence == 'geometric':
        sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                               tf.math.log(configs.config_values.sigma_low),
                                               configs.config_values.num_L))
    elif configs.config_values.sigma_sequence == 'hybrid':
        sigma_levels_geometric = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                                         tf.math.log(configs.config_values.sigma_low),
                                                         configs.config_values.num_L))
        sigma_levels_linear = tf.linspace(configs.config_values.sigma_high,
                                          configs.config_values.sigma_low,
                                          configs.config_values.num_L)
        sigma_levels = (sigma_levels_geometric + sigma_levels_linear) / 2
    return sigma_levels


def get_init_samples():
    if configs.config_values.init_samples == "":
        return None

    path = configs.config_values.init_samples
    if not os.path.exists(path):
        raise ValueError("Path ", path, " does not exist.")

    images = get_tensor_images_from_path(path)

    images /= 255
    return images


def get_tensor_images_from_path(path, resize=True):
    images = []
    for i, filename in enumerate(os.listdir(path)):
        image = tf.io.decode_image(tf.io.read_file(path + '/' + filename))
        if resize:
            size = max(image.shape[0], image.shape[1])
            is_square = image.shape[0] == image.shape[1]
            if not is_square:
                min_size = min(image.shape[0], image.shape[1])
                image = tf.image.resize_with_crop_or_pad(image, min_size, min_size)
                size = min_size
                is_square = True
            if size != 32 and is_square:
                image = tf.image.resize(image, (32, 32))
        images.append(image)

    return tf.convert_to_tensor(images)


def manage_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
