import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Multiply, concatenate, GlobalAveragePooling2D

from model.layers import RefineBlock, ConditionalFullPreActivationBlock, ConditionalInstanceNormalizationPlusPlus2D, FullPreActivationBlock

class RefineNet(keras.Model):
    def __init__(self, filters, activation, y_conditioned=False, splits=None):
        super(RefineNet, self).__init__()
        self.in_shape = None

        # Boolean is True when conditional information is concatenated to x
        self.y_conditioned = y_conditioned
        self.splits = splits

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        # FIXME: In this second preactivation block we used dilation=1, but it should have been 2
        self.preact_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)
        
        self.preact_3 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2, padding=2)
        self.preact_4 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4, padding=4)

        self.refine_block_1 = RefineBlock(activation, filters, n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=3)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        out_shape = input_shape[0][-1]
        if self.y_conditioned:
            out_shape = self.splits[0]
        
        self.decrease_channels = layers.Conv2D(out_shape, kernel_size=3, strides=1, padding='same')

    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        x = self.increase_channels(x)

        output_1 = self.preact_1([x, idx_sigmas])
        output_2 = self.preact_2([output_1, idx_sigmas])
        output_3 = self.preact_3([output_2, idx_sigmas])
        output_4 = self.preact_4([output_3, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas]) 
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output

    def summary(self):
        x = [layers.Input(name="images", shape=self.in_shape[0][1:]),
             layers.Input(name="idx_sigmas", shape=(), dtype=tf.int32)]
        return keras.Model(inputs=x, outputs=self.call(x)).summary()


class MaskedRefineNet(keras.Model):
    def __init__(self, filters, activation, splits, y_conditioned=False):
        super(MaskedRefineNet, self).__init__()
        self.in_shape = None
        self.filters = filters

        # Describes how to split the conditioning information
        # Assumes x is concatenated with its pixel-wise conditioning info
        self.splits = splits

        # Concatenate x,y too feed into RefineNet block
        # Encourages local per-pixel conditioning
        self.y_conditioned = y_conditioned

        # Filters used for every channel when dotting 
        # dotting with conditioning block
        self.conditional_embedding_size = 256

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        # FIXME: In this second preactivation block we used dilation=1, but it should have been 2
        self.preact_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)
        
        self.preact_3 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2, padding=2)
        self.preact_4 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4, padding=4)

        self.refine_block_1 = RefineBlock(activation, filters , n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=3)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None
        self.depth_extension = layers.Conv2D(self.conditional_embedding_size*self.splits[0], kernel_size=3, padding="same")

        # Added for conditioning block
        self.increase_conditional_channels = layers.Conv2D(filters, kernel_size=3, padding='same')
        # self.conditional_block = FullPreActivationBlock(activation, self.conditional_embedding_size*splits[0], kernel_size=3)
        self.conditional_block = FullPreActivationBlock(activation, self.conditional_embedding_size, kernel_size=3, dilation=2, padding=2)

        
        self.pool = GlobalAveragePooling2D(data_format="channels_last")
        self.dot  = tf.keras.layers.Dot(axes=(4, 1), name="condition_dot")



    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        old_shape = (*input_shape[0][1:-1], self.filters)
        new_shape = (*input_shape[0][1:-1], self.splits[0], self.conditional_embedding_size)
        # print("Reshaping to:", new_shape)
        self.reshape = layers.Reshape(new_shape)
        

    def call(self, inputs, mask=None):
        # xmc =  
        x_y, idx_sigmas = inputs
        x, y = tf.split(x_y, self.splits, axis=-1)
        # print("Input Shapes:", x.shape, y.shape)
        
        if self.y_conditioned:
            x = x_y

        x = self.increase_channels(x)

        output_1 = self.preact_1([x, idx_sigmas])
        output_2 = self.preact_2([output_1, idx_sigmas])
        output_3 = self.preact_3([output_2, idx_sigmas])
        output_4 = self.preact_4([output_3, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas]) 
        output = self.activation(output)
        print("RefineNet Output:", output.shape)
        output = self.depth_extension(output)
        print("Score Block:", output.shape)
        output = self.reshape(output)
        print("5D Score Block:", output.shape)

        # y = self.increase_conditional_channels(y)
        conditioning_weights = self.conditional_block(y)
        print("Condition Block:", conditioning_weights.shape)
        conditioning_weights = self.pool(conditioning_weights)
        # conditioning_weights = self.reshape(conditioning_weights)
        print("Condition Block Pooled:", conditioning_weights.shape)

        final_output = self.dot([output, conditioning_weights])
        # final_output = tf.math.multiply(output, conditioning_weights, name= "Channel_Mult")
        # final_output = tf.reduce_sum(final_output, axis=-1)

        return final_output

    def summary(self):
        x = [layers.Input(name="images", shape=self.in_shape[0][1:]),
             layers.Input(name="idx_sigmas", shape=(), dtype=tf.int32)]
        return keras.Model(inputs=x, outputs=self.call(x)).summary(line_length=100)
 

class RefineNetTwoResidual(keras.Model):

    def __init__(self, filters, activation):
        super(RefineNetTwoResidual, self).__init__()
        self.in_shape = None

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        self.preact_1_2 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        # FIXME: In this second preactivation block we used dilation=1, but it should have been 2
        self.preact_2_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3)
        self.preact_2_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)

        self.preact_3_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2,
                                                            padding=2)
        self.preact_3_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2,
                                                            padding=2)
        self.preact_4_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4,
                                                            padding=4)
        self.preact_4_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4,
                                                            padding=4)

        self.refine_block_1 = RefineBlock(activation, filters, n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=3)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[0][-1], kernel_size=3, strides=1, padding='same')

    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        x = self.increase_channels(x)

        output_1 = self.preact_1_1([x, idx_sigmas])
        output_1 = self.preact_1_2([output_1, idx_sigmas])
        output_2 = self.preact_2_1([output_1, idx_sigmas])
        output_2 = self.preact_2_2([output_2, idx_sigmas])
        output_3 = self.preact_3_1([output_2, idx_sigmas])
        output_3 = self.preact_3_2([output_3, idx_sigmas])
        output_4 = self.preact_4_1([output_3, idx_sigmas])
        output_4 = self.preact_4_2([output_4, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas])
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output

    def summary(self):
        x = [layers.Input(name="images", shape=self.in_shape[0][1:]),
             layers.Input(name="idx_sigmas", shape=(), dtype=tf.int32)]
        return keras.Model(inputs=x, outputs=self.call(x)).summary()


if __name__ == '__main__':
    import utils
    import configs
    import tensorflow_datasets as tfds

    args = utils.get_command_line_args()
    configs.config_values = args

    data_generators = tfds.load(name="mnist", split="test", batch_size=-1)
    test = tf.cast(data_generators['image'], tf.float32)

    x = test[:3]
    idx_sigmas = tf.convert_to_tensor([3, 9, 3])
    model = RefineNet(16, tf.nn.elu)
    output = model([x, idx_sigmas])
    print(model.summary())
