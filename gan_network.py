import os, time, itertools, imageio, pickle, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tools
import config
from conv_layers import *
from inception.inception_resnet_v2 import *
from tensorflow.python import pywrap_tensorflow

# leaky_relu

class ganNet():
    def __init__(self, number_of_classes, available_gpus):

        with tf.name_scope('Placeholders'):
            self.x = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, config.H_size, config.W_size, config.input_channels))
            self.z = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, 1, 1, config.noise_size))
            self.y_label = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, 1, 1, number_of_classes))
            self.y_fill = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, config.H_size, config.W_size, number_of_classes))
            self.isTrain = True
            self.G_loss_list = []
            self.D_loss_list = []
            self.G_out_list = []
            self.D_out_real_list = []
            self.D_out_fake_list = []
        
        j=0
        self.Net_collection = {}
        
        for device in available_gpus:
            with tf.device(device.name):
                with tf.variable_scope('GAN_Network') as scope:
                    if j>0:
                        scope.reuse_variables()
                    print('Building Network: ' + str(j))
                    self.build_cgan(j)
                    j = j+1

        print('Building Training: ')
        self.D_optim, self.G_optim, self.D_sum, self.G_sum, self.all_summary = self.train()


    def build_cgan(self, j):
        with tf.name_scope('Architecture'):
            # networks : generator
            G_z = self.generator(self.z[j, ...], self.y_label[j, ...], self.isTrain)

            # networks : discriminator
            D_real, D_real_logits = self.discriminator(self.x[j, ...], self.y_fill[j, ...], self.isTrain)
            D_fake, D_fake_logits = self.discriminator(G_z, self.y_fill[j, ...], self.isTrain, reuse=True)

            with tf.name_scope('Losses'):
                with tf.name_scope('D_loss_real'):
                    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([config.batch_size, D_real_logits.shape[1]])*0.99))
                with tf.name_scope('D_loss_fake'):
                    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([config.batch_size, D_fake_logits.shape[1]])+0.01))
                with tf.name_scope('D_tot_loss'):
                    new_D_loss = D_loss_real + D_loss_fake
                with tf.name_scope('G_loss'):
                    new_G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([config.batch_size, D_real_logits.shape[1]])*0.99))
                self.D_loss_list.append(new_D_loss)
                self.G_loss_list.append(new_G_loss)

            self.G_out_list.append(G_z)
            self.D_out_real_list.append(D_real)
            self.D_out_fake_list.append(D_fake)

    def lrelu(self, X, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * X + f2 * tf.abs(X)

    def generator(self, z, y_label, isTrain=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            # initializer

            # concat layer
            cat1 = tf.concat([z, y_label], 3)

            # conv_1 = resnet_block(cat1, 512, 2)
            # conv_2 = resnet_block(conv_1, 512, 2)
            # conv_3 = resnet_block(conv_2, 256, 4)
            # conv_4 = resnet_block(conv_3, 128, 4)
            # conv_5 = resnet_block(conv_1, 64, 8)
            # conv_6 = resnet_block(conv_5, 3, 2)
            # conv_7 = resnet_block(conv_6, 64, 4)
            # conv_8 = resnet_block(conv_7, config.input_channels, 4)

            conv_1 = double_deconv(cat1, 512)
            # conv_2 = double_deconv(conv_1, 512)
            # conv_3 = double_deconv(conv_2, 256)
            conv_4 = double_deconv(conv_1, 512)
            conv_5 = double_deconv(conv_4, 256)
            conv_6 = double_deconv(conv_5, 128)
            conv_7 = double_deconv(conv_6, 64)
            conv_8 = same_size_conv(conv_7, 16)
            conv_8 = same_size_conv(conv_8, 8)
            conv_8 = same_size_conv(conv_8, 4)
            conv_8 = same_size_conv(conv_8, 2)
            conv_8 = double_deconv(conv_8, 32, do_BN=True)
            conv_8 = same_size_conv(conv_8, 32)
            conv_8 = same_size_conv(conv_8, 16)
            conv_8 = same_size_conv(conv_8, 8)
            conv_8 = same_size_conv(conv_8, 4)
            conv_8 = same_size_conv(conv_8, 2)
            conv_8 = double_deconv(conv_8, config.input_channels, do_BN=False)


            # # 1st hidden layer
            # deconv1 = tf.layers.conv2d_transpose(cat1, 256, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv1, training=isTrain), alpha=0.2)
            # print(lrelu1)

            # # 2nd hidden layer
            # deconv2 = tf.layers.conv2d_transpose(lrelu1, 512, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv2, training=isTrain), alpha=0.2)
            # print(lrelu2)

            # # 3nd hidden layer
            # deconv3 = tf.layers.conv2d_transpose(lrelu2, 512, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv3, training=isTrain), alpha=0.2)
            # conv3 = tf.layers.conv2d(lrelu3, 512, [5, 5], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain), alpha=0.2)
            # print(lrelu3)
            
            # # 4nd hidden layer
            # deconv4 = tf.layers.conv2d_transpose(lrelu3, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv4, training=isTrain), alpha=0.2)
            # conv4 = tf.layers.conv2d(lrelu4, 256, [5, 5], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain), alpha=0.2)
            # print(lrelu4)

            # # 5nd hidden layer
            # deconv5 = tf.layers.conv2d_transpose(lrelu4, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu5 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv5, training=isTrain), alpha=0.2)
            # conv5 = tf.layers.conv2d(lrelu5, 128, [5, 5], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu5 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv5, training=isTrain), alpha=0.2)

            # # # 6nd hidden layer
            # # deconv6 = tf.layers.conv2d_transpose(lrelu5, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # # lrelu6 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv6, training=isTrain), alpha=0.2)

            # # output layer
            # deconv7 = tf.layers.conv2d_transpose(lrelu5, config.input_channels, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            o = tf.nn.tanh(conv_8)

            return o

    def discriminator(self, x, y_fill, isTrain=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # cat1 = tf.concat([x, y_fill], 3)
            # net, endpoint = inception_resnet_v2(cat1, is_training=False)
            # inception_out = endpoint['Logits']
            # initializer
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            # tensor_std = tf.math.reduce_std(x, axis=0)
            # tensor_std = tf.math.reduce_mean(tensor_std)

            # concat layer
            x = tf.keras.layers.GaussianNoise(0.01)(x)

            # std_channel = y_fill*0 + tensor_std
            # cat1 = tf.concat([x, y_fill, std_channel], 3)
            cat1 = tf.concat([x, y_fill], 3)

            conv_256 = half_conv(cat1, 256, 5)
            conv_256 = same_size_conv(conv_256, 5)
            conv_256 = same_size_conv(conv_256, 5)
            conv_128 = half_conv(conv_256, 128, 5)
            conv_128 = same_size_conv(conv_128, 5)
            conv_128 = same_size_conv(conv_128, 5)
            # conv_64 = half_conv(conv_128, 128, 5)
            # conv_32 = half_conv(conv_64, 64, 5)
            # conv_32 = same_size_conv(conv_32, 5)
            # conv_16 = half_conv(conv_32, 32, 5)
            # conv_8 = half_conv(conv_16, 32, 5)
            # conv_4 = half_conv(conv_8, 16, 5)
            conv_2 = half_conv(conv_128, 64, 2, do_BN=False)
            # activated = tf.math.tanh(conv_2)
            # # 1st hidden layer
            # conv1 = tf.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu1 = self.lrelu(conv1, 0.2)

            # # 2nd hidden layer
            # conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            # lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            # # output layer
            # conv3 = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
            # reshaped_conv_out = tf.reshape(conv_4, [config.batch_size, shape_conv[1]*shape_conv[2]*shape_conv[3]])
            logit = tf.layers.dense(conv_2, 1)
            # logit = conv_2
            print(logit)
            shape_logit = logit.shape
            logit = tf.reshape(logit, [config.batch_size, shape_logit[1]*shape_logit[2]*shape_logit[3]])
            o = tf.nn.sigmoid(logit)

            return o, logit

    def train(self):
        with tf.name_scope('Other_Operations'):
            with tf.name_scope('Training_operation'):
                global_step = tf.Variable(0, trainable=False)

                lr_G = tf.compat.v1.train.exponential_decay(0.001, global_step, 500, 0.99, staircase=False)
                lr_D = tf.compat.v1.train.exponential_decay(0.001, global_step, 500, 0.99, staircase=False)

                D_loss = sum(self.D_loss_list)
                G_loss = sum(self.G_loss_list)

                # trainable variables for each network
                T_vars = tf.trainable_variables()
                D_vars = [var for var in T_vars if 'discriminator' in var.name]
                G_vars = [var for var in T_vars if 'generator' in var.name ]

                # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optim = tf.train.AdamOptimizer(lr_D, beta1=0.5)
                D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
                G_optim = tf.train.AdamOptimizer(lr_G, beta1=0.5).minimize(G_loss, var_list=G_vars)

            with tf.name_scope('Summaries_discriminator'):
                D_loss_sum = tf.summary.scalar('D_loss', D_loss)
                G_loss_sum = tf.summary.scalar('G_loss', G_loss)
                Lr_G = tf.summary.scalar('lr_G', lr_G)
                Lr_D = tf.summary.scalar('lr_D', lr_D)
                D_Fake = tf.reduce_mean(tf.concat(self.D_out_fake_list, axis = 0))
                D_Real = tf.reduce_mean(tf.concat(self.D_out_real_list, axis = 0))
                Optimal_distance = (D_Real - D_Fake)
                D_Fake_sum = tf.summary.scalar("D_Fake", D_Fake)
                D_Real_sum = tf.summary.scalar("D_Real", D_Real)
                Optimal_distance_sum = tf.summary.scalar("Optimal_distance", Optimal_distance)
                Gen_pics_sum = tf.summary.image('Generated_pics', tf.concat(self.G_out_list, axis=0), max_outputs=3)
                shape_real = self.x.shape
                Real_pics = tf.reshape(self.x, [shape_real[0]*shape_real[1], shape_real[2], shape_real[3], shape_real[4]])

                Gen_pics_sum = tf.summary.image('Real_pics', tf.concat(Real_pics, axis=0), max_outputs=3)

                Discriminator_summary = tf.summary.merge([Optimal_distance_sum, D_loss_sum, G_loss_sum, D_Fake_sum, D_Real_sum, Lr_D])
                Generator_summary = tf.summary.merge([G_loss_sum, D_Fake_sum, Gen_pics_sum, Lr_G])
                all_summary = tf.summary.merge_all()


            with tf.name_scope('Model_Saver'):
                self.model_saver = tf.train.Saver()

            # with tf.name_scope('Inception_Loader'): 
            #         variables = tf.contrib.slim.get_variables_to_restore()
            #         ckpt_path = './inception/inception_resnet_v2_2016_08_30.ckpt'
            #         reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
            #         vars_in_checkpoint = reader.get_variable_to_shape_map()
            #         inception_in_Net_variable = [v for v in variables if 'InceptionResnetV2' in v.name.split(':')[0]]
            #         name_to_vars = {''.join(v.op.name.split('GAN_Network/discriminator/')[0:]): v for v in inception_in_Net_variable}
            #         map_name = {}
            #         for el in name_to_vars:
            #             if el in vars_in_checkpoint:
            #                 if vars_in_checkpoint[el] == list(name_to_vars[el].shape):
            #                     map_name[el] = name_to_vars[el]
            #         self.inception_loader = tf.train.Saver(var_list=map_name)

            with tf.name_scope('prev_checkpoint_loader'):
                    if os.path.isfile('./checkpoint/checkpoint'):
                        ckpts = tf.train.latest_checkpoint('./checkpoint')
                        vars_in_checkpoint = tf.train.list_variables(ckpts)
                        variables = tf.contrib.slim.get_variables_to_restore()
                        ckpt_var_name = []
                        ckpt_var_shape = {}
                        for el in vars_in_checkpoint:
                            ckpt_var_name.append(el[0])
                            ckpt_var_shape[el[0]] = el[1]
                        var_list = [v for v in variables if v.name.split(':')[0] in ckpt_var_name]
                        var_list = [v for v in var_list if list(v.shape) == ckpt_var_shape[v.name.split(':')[0]]]
                        self.prev_checkpoint_loader = tf.train.Saver(var_list=var_list)
        
            with tf.name_scope("Initializer"):
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                self.init = tf.group(init_local, init_global)

        return D_optim, G_optim, Discriminator_summary, Generator_summary, all_summary
