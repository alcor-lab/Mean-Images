import tensorflow as tf
from math import ceil

w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
b_init = tf.constant_initializer(0.0)

def double_deconv(input_tensor, channels, do_BN=True):
    with tf.name_scope('double_deconv'):

        input_shape = input_tensor.get_shape().as_list()

        if input_shape[1] == 1:
            stride_h = 1
            stride_w = 1
        else:
            stride_h = 2
            stride_w = 2
        kernel = [2, 2]

        out = tf.layers.conv2d_transpose(input_tensor, channels, kernel, strides=(stride_h, stride_w), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        if do_BN:
            batch_normalization = tf.layers.batch_normalization(out, training=True)
        else:
            batch_normalization = out
        lrelu = tf.nn.leaky_relu(batch_normalization, alpha=0.2)

        return lrelu

def upsampling_branch(input_tensor, channels):
    with tf.name_scope('upsampling_branch'):
        kernel = [1, 1]
        stride_h = 2
        stride_w = 2
        out = tf.layers.conv2d_transpose(input_tensor, channels, kernel, strides=(stride_h, stride_w), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
    return out


def same_size_conv(input_tensor, k_size, do_BN=True, out_channels=None):
    with tf.name_scope('same_size_conv'):
        input_shape = input_tensor.get_shape().as_list()
        if out_channels == None:
            channels = input_shape[-1]
        else:
            channels = out_channels
            
        conv = tf.layers.conv2d(input_tensor, channels, [k_size, k_size], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        if do_BN:
            batch_normalization = tf.layers.batch_normalization(conv, training=True)
        else:
            batch_normalization = conv
        lrelu = tf.nn.leaky_relu(batch_normalization, alpha=0.2)

    return lrelu

def resnet_block(input_tensor, channels, k_size):
    with tf.name_scope('residual_block'):

        # tensorscale = tf.random_uniform([1], 0, 1)
        # scale = tf.Variable(initial_value=tensorscale)

        out_same = same_size_conv(input_tensor, k_size)
        # double = double_deconv(out_same, channels)
        # residual = upsampling_branch(input_tensor, channels)

        # add = scale*residual + (1-scale)*double
        add = input_tensor + out_same
        batch_normalization = tf.layers.batch_normalization(add, training=True)
        lrelu = tf.nn.leaky_relu(batch_normalization, alpha=0.2)

    return lrelu

def half_conv(input_tensor, channels, kernel, do_BN=True):
    with tf.name_scope('double_deconv'):

        input_shape = input_tensor.get_shape().as_list()

        if input_shape[1] == 1:
            stride_h = 1
            stride_w = 1
        else:
            stride_h = 2
            stride_w = 2
        kernel = [kernel, kernel]

        out = tf.layers.conv2d(input_tensor, channels, kernel, strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        if do_BN:
            batch_normalization = tf.layers.batch_normalization(out, training=True)
        else:
            batch_normalization = out
        lrelu = tf.nn.leaky_relu(batch_normalization, alpha=0.2)

        return lrelu
