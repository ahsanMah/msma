import tensorflow as tf
import tensorflow_probability as tfp

@tf.function(experimental_compile=True)
def normalized_dsm_loss(score, x_perturbed, x, sigmas, masks):
    target = (x_perturbed - x) / (tf.square(sigmas))
    loss = 0.5 * tf.reduce_sum(tf.square(score+target), axis=[1,2,3], keepdims=True) * tf.square(sigmas)
    loss /= tf.reduce_sum(masks, axis=[1,2], keepdims=True)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function(experimental_compile=True)
def dsm_loss(score, x_perturbed, x, sigmas):
    target = (x_perturbed - x) / (tf.square(sigmas))
    loss = 0.5 * tf.reduce_sum(tf.square(score+target), axis=[1,2,3], keepdims=True) * tf.square(sigmas)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def ssm_loss(score_net, data_batch):
    sum_over = list(range(1, len(data_batch.shape)))
    v = tf.random.normal(data_batch.shape)
    with tf.GradientTape() as t:
        t.watch(data_batch)
        scores = score_net(data_batch)
        v_times_scores = tf.reduce_sum(v * scores)
    grad = t.gradient(v_times_scores, data_batch)
    loss_first = tf.reduce_sum(grad * v, axis=sum_over)
    loss_second = 0.5 * tf.reduce_sum(tf.square(scores), axis=sum_over)
    loss = tf.reduce_mean(loss_first + loss_second)

    return loss

# Initial radius
NU = 0.1

@tf.function
def update_radius(radii):
    return tfp.stats.percentile(radii, q=100 * NU)
    

@tf.function
def ocnn_loss(r, w, V, y_pred):
    
    term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
    term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
    term3 = 1 / NU * tf.reduce_mean(tf.maximum(
                                    0.0,r - tf.reduce_max(y_pred, axis=1)), axis=-1)
    term4 = -1*r

    return (term1 + term2 + term3 + term4)
