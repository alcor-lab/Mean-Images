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
import pprint
pp = pprint.PrettyPrinter(indent=4)

# leaky_relu

class ganNet():
    def __init__(self, number_of_classes, available_gpus, mean_values, std_values):

        self.number_of_classes = number_of_classes
        with tf.name_scope('Placeholders'):
            self.x = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, config.H_size, config.W_size, config.input_channels))
            self.x_1 = self.x*2-1
            self.z = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, config.noise_size))
            self.y_label = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, 1, 1, number_of_classes + 1))
            self.y_label_squeezed = tf.squeeze(self.y_label)
            self.y_label_id = tf.math.argmax(self.y_label_squeezed, axis=-1)
            self.y_fill = tf.placeholder(tf.float32, shape=(len(available_gpus), config.batch_size, config.num_patches, config.num_patches, number_of_classes + 1))
            self.mean_values = tf.convert_to_tensor(mean_values, np.float32)
            self.std_values = tf.convert_to_tensor(std_values, np.float32)
            self.isTrain = True

            gen_one_hot = tf.one_hot(indices=[self.number_of_classes], depth=self.number_of_classes + 1)
            multiply = tf.constant([config.batch_size, config.num_patches, config.num_patches, 1])
            gen_one_hot = tf.expand_dims(gen_one_hot, axis=1)
            gen_one_hot = tf.expand_dims(gen_one_hot, axis=1)
            self.gen_one_hot = tf.tile(gen_one_hot, multiply)
            print(self.gen_one_hot)

            self.summaries_dictionary = {}
        
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
        self.D_optim, self.G_optim, self.all_summary = self.train()

        pp.pprint(self.summaries_dictionary)


    def build_cgan(self, j):
        with tf.name_scope('Architecture'):
            # networks : generator
            # with tf.name_scope('Generators'):
            conv_input, cond_embedding = self.conv_input_generator(self.z[j, ...], self.y_label_squeezed[j, ...],'8', 64*config.Ng, self.isTrain)
            G_64_out, G_64 = self.generator_first(conv_input, '64', 4*config.Ng, self.isTrain)
            G_128_out, G_128 = self.generator_other(cond_embedding, G_64,'128', 2*config.Ng, self.isTrain)
            G_256_out, G_256 = self.generator_other(cond_embedding, G_128,'256', config.Ng, self.isTrain)

            # with tf.name_scope('Input_resizes'):
            x_64 = tf.image.resize(self.x_1[j, ...], [64,64], name='x_64')
            x_128 = tf.image.resize(self.x_1[j, ...], [128,128], name='x_128')
            x_256 = self.x_1[j, ...]

            y_fill = self.y_fill[j, ...]

            y_label_id = self.y_label_id[j, ...]

            # networks : discriminator
            # with tf.name_scope('Discriminators'):
            with tf.name_scope('Discriminator_64'):G_64_out
                D_real_logits_64, D_real_64, D_real_class_logit_64 , D_real_class_64 = self.discriminator(G_64_out, y_fill, 3, 'discriminator_64x64', 8*config.Nd, self.isTrain, reuse=False)
                D_fake_logits_64, D_fake_64, D_fake_class_logit_64 , D_fake_class_64 = self.discriminator(x_64, y_fill, 3, 'discriminator_64x64', 8*config.Nd, self.isTrain, reuse=True)

            with tf.name_scope('Discriminator_128'):
                D_real_logits_128, D_real_128, D_real_class_logit_128 , D_real_class_128 = self.discriminator(x_128, y_fill, 4, 'discriminator_128x128', 8*config.Nd, self.isTrain, reuse=False)
                D_fake_logits_128, D_fake_128, D_fake_class_logit_128 , D_fake_class_128 = self.discriminator(G_128_out, y_fill, 4, 'discriminator_128x128', 8*config.Nd, self.isTrain, reuse=True)

            with tf.name_scope('Discriminator_256'):
                D_real_logits_256, D_real_256, D_real_class_logit_256 , D_real_class_256 = self.discriminator(x_256, y_fill, 5, 'discriminator_256x256', 8*config.Nd, self.isTrain, reuse=False)
                D_fake_logits_256, D_fake_256, D_fake_class_logit_256 , D_fake_class_256 = self.discriminator(G_256_out, y_fill, 5, 'discriminator_256x256', 8*config.Nd, self.isTrain, reuse=True)

            with tf.name_scope('Losses'):

                with tf.name_scope('color_losses'):
                    # tf.stats.covariance
                    color_mean_64 = tf.reduce_mean(G_64_out, axis=[1,2])
                    color_std_64 = tf.math.reduce_std(G_64_out, axis=[1,2])
                    color_mean_128 = tf.reduce_mean(G_128_out, axis=[1,2])
                    color_std_128 = tf.math.reduce_std(G_128_out, axis=[1,2])
                    color_mean_256 = tf.reduce_mean(G_256_out, axis=[1,2])
                    color_std_256 = tf.math.reduce_std(G_256_out, axis=[1,2])

                    G_128_color_loss = self.Loss_color(color_mean_128, color_mean_64, color_std_128, color_std_64)
                    G_256_color_loss = self.Loss_color(color_mean_256, color_mean_128, color_std_256, color_std_128)

                D_loss_64, D_tot_loss_class_64, G_loss_64, G_loss_class_64, G_intra_class_loss_64 = self.Loss_compute('64', D_real_logits_64, D_fake_logits_64, y_fill, D_real_class_logit_64, D_fake_class_logit_64, G_64_out, y_label_id)
                D_loss_128, D_tot_loss_class_128, G_loss_128, G_loss_class_128, G_intra_class_loss_128 = self.Loss_compute('128', D_real_logits_128, D_fake_logits_128, y_fill, D_real_class_logit_128, D_fake_class_logit_128, G_128_out, y_label_id)
                D_loss_256, D_tot_loss_class_256, G_loss_256, G_loss_class_256, G_intra_class_loss_256 = self.Loss_compute('256', D_real_logits_256, D_fake_logits_256, y_fill, D_real_class_logit_256, D_fake_class_logit_256, G_256_out, y_label_id)

                with tf.name_scope('Tot_loss'):

                    D_tot_GAN = D_loss_64 + D_loss_128 + D_loss_256
                    D_tot_Classification = config.alfa_classification_D*(D_tot_loss_class_64 + D_tot_loss_class_128 + D_tot_loss_class_256) 

                    G_tot_GAN = G_loss_64 + G_loss_128 + G_loss_256 
                    G_tot_Color = config.alfa_color*(G_128_color_loss + G_256_color_loss)
                    G_tot_Classification = config.alfa_classification_G*(G_loss_class_64 + G_loss_class_128 + G_loss_class_256)
                    G_tot_intra_class_loss = config.alfa_mean*(G_intra_class_loss_64 + G_intra_class_loss_128 + G_intra_class_loss_256)

                    D_tot_loss = D_tot_GAN + D_tot_Classification
                    G_tot_loss = G_tot_GAN + G_tot_Color + G_tot_Classification + G_tot_intra_class_loss

            self.add_summary_dict('G_color_loss_', ['128', '256'], [G_128_color_loss, G_256_color_loss], type_summary='scalar')
            self.add_summary_dict('G_out_', ['64', '128', '256'], [(G_64_out+1)/2, (G_128_out+1)/2, (G_256_out+1)/2], type_summary='image')
            self.add_summary_dict('D_prob_fake_', ['64', '128', '256'], [D_fake_64, D_fake_128, D_fake_256], type_summary='scalar')
            self.add_summary_dict('D_prob_real_', ['64', '128', '256'], [D_real_64, D_real_128, D_real_256], type_summary='scalar')
            self.add_summary_dict('D_tot_GAN_loss', [''], [D_tot_GAN], type_summary='scalar')
            self.add_summary_dict('D_tot_Classification_loss', [''], [D_tot_Classification], type_summary='scalar')
            self.add_summary_dict('G_tot_GAN_loss', [''], [G_tot_Color], type_summary='scalar')
            self.add_summary_dict('G_tot_Classification_loss', [''], [G_tot_Classification], type_summary='scalar')
            self.add_summary_dict('G_tot_intra_class_loss', [''], [G_tot_intra_class_loss], type_summary='scalar')
            self.add_summary_dict('D_tot_loss', [''], [D_tot_loss], type_summary='scalar')
            self.add_summary_dict('G_tot_loss', [''], [G_tot_loss], type_summary='scalar')

    def Loss_real(self, real_logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones([config.batch_size, real_logits.shape[1]])*0.99))

    def Loss_fake(self, fake_logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros([config.batch_size, fake_logits.shape[1]])+0.01))

    def Loss_color(self, color_mean_up, color_mean_down, color_std_up, color_std_down):
        #Frobenius norm
        L_mean = tf.norm(color_mean_up - tf.stop_gradient(color_mean_down))
        L_std = tf.norm(color_std_up - tf.stop_gradient(color_std_down))

        return tf.reduce_mean(config.lambda_1*L_mean + config.lambda_2*L_mean)

    def Loss_classification(self,one_hot_label, discriminator_logit):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=discriminator_logit))

    def intra_class_loss(self,g_o, y_label_id):
        with tf.name_scope('intra_class_loss'):
            Loss = 0
            n = 0
            shape_g_o = g_o.get_shape().as_list()
            for j in range(shape_g_o[0]):
                one_output = g_o[j, :, :]
                id_class = y_label_id[j]
                mean_i = self.mean_values[id_class, :, :, :]
                mean_i = tf.image.resize(mean_i, [shape_g_o[1],shape_g_o[2]])
                diff =  one_output - mean_i
                diff = tf.image.resize(diff, [config.mean_size,config.mean_size])
                for c in range(shape_g_o[-1]):

                    S_inv_i = self.std_values[id_class, :, :, c]

                    mult = tf.matmul(tf.transpose(diff[:, :, c], perm=[1,0]), S_inv_i, transpose_a = True)
                    mult = tf.matmul(mult, diff[:, :, c])
                    Loss += tf.norm(mult)
            Loss = Loss
        return Loss

    def mean_std_mse(self,g_o, y_label_id):
        with tf.name_scope('intra_class_loss'):
            dist_list = []
            shape_g_o = g_o.get_shape().as_list()
            for j in range(shape_g_o[0]):
                one_output = (g_o[j, :, :] + 1)/2
                id_class = y_label_id[j]
                mean_i = self.mean_values[id_class, :, :, :]
                mean_i = tf.image.resize(mean_i, [shape_g_o[1],shape_g_o[2]])
                dist = tf.math.square(one_output - mean_i)
                dist = tf.expand_dims(dist, axis=0)
                dist_list.append(dist)

        Loss = tf.reduce_mean(tf.reduce_mean(tf.concat(dist_list, axis=0), axis=0))
            
        return Loss
        
    def Loss_compute(self, name, D_real_logits, D_fake_logits, y_fill, D_real_class_logit, D_fake_class_logit, g_out, y_label_id):
        with tf.name_scope('Losses_' + name):
            with tf.name_scope('D_loss_real_' + name):
                D_loss_real =  self.Loss_real(D_real_logits)
            with tf.name_scope('D_loss_fake_' + name):
                D_loss_fake =  self.Loss_fake(D_fake_logits)
            with tf.name_scope('D_tot_loss_' + name):
                D_loss = D_loss_real + D_loss_fake

            with tf.name_scope('D_loss_real_class_' + name):
                D_loss_real_class =  self.Loss_classification(y_fill, D_real_class_logit)
            with tf.name_scope('D_loss_fake_class_' + name):
                D_loss_fake_class =  self.Loss_classification(self.gen_one_hot, D_fake_class_logit)
            with tf.name_scope('D_tot_loss_class_' + name):
                D_tot_loss_class = D_loss_real_class + D_loss_fake_class

            with tf.name_scope('G_loss_' + name):
                G_loss =  self.Loss_real(D_fake_logits)
            with tf.name_scope('G_loss_class_' + name):
                G_loss_class =  self.Loss_classification(y_fill, D_fake_class_logit)

            with tf.name_scope('G_intra_class_loss' + name):
                # G_intra_class_loss = self.mean_std_mse(g_out, y_label_id)
                G_intra_class_loss = self.intra_class_loss(g_out, y_label_id)

            self.add_summary_dict('G_loss_', [name], [G_loss], type_summary='scalar')
            self.add_summary_dict('G_loss_class_', [name], [G_loss_class], type_summary='scalar')
            self.add_summary_dict('D_tot_loss_class_', [name], [D_tot_loss_class], type_summary='scalar')
            self.add_summary_dict('G_intra_class_loss_', [name], [G_intra_class_loss], type_summary='scalar')
                
        return D_loss, D_tot_loss_class, G_loss, G_loss_class, G_intra_class_loss

    def add_summary_dict(self, var_name, name_range, var_list, type_summary=None):
        i = 0
        for number in name_range:
            var_name_new = var_name
            if number != '':
                var_name_new = number + 'x' + number + '/' + var_name + number
            else:
                var_name_new = 'all/' + var_name

            if 'color' in var_name:
                var_name_new = 'color/' + var_name

            variable = var_list[i]
            i += 1
            if var_name_new not in self.summaries_dictionary:
                self.summaries_dictionary[var_name_new] = {}
                self.summaries_dictionary[var_name_new]['list'] = [variable]
                self.summaries_dictionary[var_name_new]['type'] = type_summary
            else:
                self.summaries_dictionary[var_name_new]['list'].append(variable)

    def conv_input_generator(self, z, y_label, name, Ng, isTrain=True, reuse=False):
        with tf.variable_scope('conv_input_generator', reuse=reuse):
            # concat layer
            cond_embedding = tf.layers.dense(y_label, config.cond_emb_dim)
            cat1 = tf.concat([z, cond_embedding], 1)

            cat1 = tf.layers.dense(cat1, Ng*4*4)

            cat1 = tf.reshape(cat1, [config.batch_size, 4,4, Ng])

            return cat1, cond_embedding

    def generator_first(self, conv_input, name, Ng, isTrain=True, reuse=False):
        with tf.variable_scope('generator_first', reuse=reuse):
            # concat layer

            conv_8 = double_deconv(conv_input, 8*Ng)
            conv_16 = double_deconv(conv_8,  4*Ng)
            conv_32 = double_deconv(conv_16,  2*Ng)
            conv_64 = double_deconv(conv_32,  Ng)
            conv_out = same_size_conv(conv_64, 3, do_BN=False, out_channels=config.input_channels)

            out = tf.nn.tanh(conv_out)

            return out, conv_64

    def generator_other(self, y_label,prev_conv, name, Ng, isTrain=True, reuse=False):
        with tf.variable_scope('generator_other_' + name, reuse=reuse):
            prev_conv = tf.stop_gradient(prev_conv)
            shape_prev_conv = prev_conv.get_shape().as_list()

            # concat layer
            multiply = tf.constant([1, shape_prev_conv[1], shape_prev_conv[2], 1])
            y_label = tf.expand_dims(y_label, axis=1)
            y_label = tf.expand_dims(y_label, axis=1)
            matrix = tf.tile(y_label, multiply)
            cat2 = tf.concat([prev_conv, matrix], 3)

            res_1 = resnet_block(cat2, shape_prev_conv[-1], 3)
            res_2 = resnet_block(res_1, shape_prev_conv[-1], 3)
            conv_double = double_deconv(res_2, Ng, do_BN=True)
            conv_out = same_size_conv(conv_double, 3, do_BN=False, out_channels=config.input_channels)

            out = tf.nn.tanh(conv_out)

            return out, conv_double

    def discriminator(self, x, y_fill, number_of_block, name, Ng, isTrain=True, reuse=False):
        with tf.variable_scope(name, reuse=reuse):

            x = tf.keras.layers.GaussianNoise(0.01)(x)

            # cat1 = tf.concat([x, y_fill], 3)

            half_tensor = x
            channels = np.linspace(16,Ng,number_of_block)
            channels = [int(c) for c in channels]
            for n in range(number_of_block):
                with tf.name_scope('half_conv_' + str(n)):
                    channel = channels[n]
                    if n == number_of_block-1:
                        half_tensor = half_conv(half_tensor, channel, 3, do_BN=False)
                    else:
                        half_tensor = half_conv(half_tensor, channel, 3)


            with tf.name_scope('Discriminative'):
                cat2 = tf.concat([half_tensor, y_fill], 3)
                prob_logit = tf.layers.dense(cat2, 1)
                shape_prob_logit = prob_logit.shape
                prob_logit = tf.reshape(prob_logit, [config.batch_size, shape_prob_logit[1]*shape_prob_logit[2]*shape_prob_logit[3]])
                prob = tf.nn.sigmoid(prob_logit)
            with tf.name_scope('Classificative'):
                classes_logit = tf.layers.dense(half_tensor, self.number_of_classes + 1)
                classes = tf.nn.softmax(classes_logit)
            return prob_logit, prob, classes_logit, classes
    
    def multi_class_KFD(self,name):
        with tf.name_scope('KFD'):
            var_name = name + 'x' + name + '/G_out_' + name
            tensor_out = self.summaries_dictionary[var_name]['list']

            reshaped_tensor_out = [tf.expand_dims(t, axis=0) for t in tensor_out]
            y = tf.concat(reshaped_tensor_out, axis=0)
            mean = tf.reduce_mean(y, axis=0)

            all_mean = tf.reduce_mean(y, axis=0)

            shape_mean = mean.get_shape().as_list()

            S_B = 0
            for i in range(shape_mean[0]):
                diff = mean[i] -  all_mean
                mult = tf.matmul(diff, diff, transpose_b = True)
                S_B += mult

            shape_y = mean[0].get_shape().as_list()
            S_W = 0
            for i in range(shape_mean[0]):
                for j in range(shape_y[0]):
                    one_output = y[j, i, ...]
                    mean_i =   mean[i]
                    diff =  one_output - mean_i
                    mult = tf.matmul(diff, diff, transpose_b = True)
                S_W += mult

        return S_W, S_B



    def train(self):
        with tf.name_scope('Other_Operations'):
            with tf.name_scope('prev_checkpoint_loader'):
                    if os.path.isfile(config.checkpoint_file_check):
                        ckpts = tf.train.latest_checkpoint(config.ckpt_load_folder)
                        vars_in_checkpoint = tf.train.list_variables(ckpts)
                        variables = tf.contrib.slim.get_variables_to_restore()
                        ckpt_var_name = []
                        ckpt_var_shape = {}
                        for el in vars_in_checkpoint:
                            ckpt_var_name.append(el[0])
                            ckpt_var_shape[el[0]] = el[1]
                        # var_list = [v for v in variables if v.name.split(':')[0] in ckpt_var_name]
                        var_list = [v for v in variables if v.name.split(':')[0] in ckpt_var_name and ('lr_D' not in v.name.split(':'[0]) and 'lr_G' not in v.name.split(':'[0]))]
                        # pp.pprint(var_list)
                        var_list = [v for v in var_list if list(v.shape) == ckpt_var_shape[v.name.split(':')[0]]]
                        self.prev_checkpoint_loader = tf.train.Saver(var_list=var_list)
                        
            with tf.name_scope('Training_operation'):
                global_step = tf.Variable(0, trainable=False)

                lr_G = tf.compat.v1.train.exponential_decay(0.001, global_step, 1000, 0.99, staircase=False, name='G_lr')
                lr_D = tf.compat.v1.train.exponential_decay(0.001, global_step, 1000, 0.99, staircase=False, name='D_lr')

                D_loss = sum(self.summaries_dictionary['all/D_tot_loss']['list'])
                G_loss = sum(self.summaries_dictionary['all/G_tot_loss']['list'])

                # trainable variables for each network
                T_vars = tf.trainable_variables()
                D_vars = [var for var in T_vars if 'discriminator' in var.name]
                G_vars = [var for var in T_vars if 'generator' in var.name ]

                # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optim = tf.train.AdamOptimizer(lr_D, beta1=0.5)
                D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
                G_optim = tf.train.AdamOptimizer(lr_G, beta1=0.5).minimize(G_loss, var_list=G_vars)

            

            with tf.name_scope('Summaries_discriminator'):
                for name in self.summaries_dictionary.keys():
                    tensor_list = self.summaries_dictionary[name]['list']
                    summary_type = self.summaries_dictionary[name]['type']
                    if summary_type == 'scalar': 
                        shape = tensor_list[0].get_shape().as_list()
                        if len(shape) >0:
                            value = tf.reduce_mean(tf.concat(tensor_list, axis = 0))
                        else:
                            value = tf.reduce_mean(tensor_list)
                        tf.summary.scalar(name, value)
                    elif summary_type == 'image':
                        tf.summary.image(name, tf.concat(tensor_list, axis=0), max_outputs=1)

                tf.summary.scalar('all_net/lr_G', lr_G)
                tf.summary.scalar('all_net/lr_D', lr_D)
                shape_real = self.x.shape
                Real_pics = tf.reshape(self.x, [shape_real[0]*shape_real[1], shape_real[2], shape_real[3], shape_real[4]])
                tf.summary.image('Real_pics', tf.concat(Real_pics, axis=0), max_outputs=3)

                all_summary = tf.summary.merge_all()


            with tf.name_scope('Model_Saver'):
                self.model_saver = tf.train.Saver()

            
        
            with tf.name_scope("Initializer"):
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                self.init = tf.group(init_local, init_global)

        return D_optim, G_optim, all_summary
