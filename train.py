import os
import config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config.visible_gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stack_gan_network import ganNet
import tensorflow.compat.v1 as tf
import pprint
import datetime
from batch_generator import IO_manager
import tools as tools
from tqdm import tqdm
from datetime import datetime
import numpy as np
import cv2

pp = pprint.PrettyPrinter(indent=4)
tf_config = tf.ConfigProto(inter_op_parallelism_threads=config.inter_op_parallelism_threads, allow_soft_placement = True)
tf_config.gpu_options.allow_growth = config.allow_growth

def train():
    with tf.Session(config=tf_config) as sess:

        available_gpus = tools.get_available_gpus()
        Io_tool = IO_manager()
        print(Io_tool.id_to_label)

        network = ganNet(Io_tool.number_of_classes, available_gpus, Io_tool.mean, Io_tool.std_inverse)

        #create new log
        now = datetime.now()
        date_time = now.strftime("%m-%d, %H:%M")
        log_writer = tf.summary.FileWriter("logdir/" + config.action + '/' + date_time + "/train", sess.graph)
        
        sess.run(network.init)

        # Loading initial presaved network
        if os.path.isfile(config.checkpoint_file_check) and config.load_previous_weigth:
            print('Loading from checkpoints')
            ckpts = tf.train.latest_checkpoint(config.ckpt_load_folder)
            network.prev_checkpoint_loader.restore(sess, ckpts)
        # else:
        #     if config.load_pretrained_inception:
        #         print('load_pretrained_inception')
        #         ckpt_path = './inception/inception_resnet_v2_2016_08_30.ckpt'
        #         network.inception_loader.restore(sess, ckpt_path)

        step = 0
        
        def show_result():
            ready_batch_show = Io_tool.compute_batch(Devices=len(available_gpus), Train_type='Generator')
            for batch in ready_batch_show:
                placeholder_dict = {network.z: batch['Z'],
                                    network.y_label: batch['Y_label']}

                list_run = network.summaries_dictionary['256x256/G_out_256']['list'] +
                            network.summaries_dictionary['128x128/G_out_128']['list'] +
                            network.summaries_dictionary['64x64/G_out_64']['list']

                test_images = sess.run(list_run , placeholder_dict)

                class_labels = np.argmax(batch['Y_label'], axis=4) 

                for j in range(len(test_images)):
                    image_device = test_images[j]
                    labels_device = class_labels[j, ...]
                    for i in range(image_device.shape[0]):
                        image = image_device[i, :, :, :]
                        norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # norm_image = (((image + 1)/2)*255).astype(int)
                        class_id = labels_device[i, 0, 0]
                        activity_name = Io_tool.id_to_label[int(class_id)]

                        path = config.out_path + '/' + activity_name
                        if not os.path.exists(path):
                            os.makedirs(path)

                        counter = 0
                        filename = path + "/" + activity_name + "_result{}.png"
                        while os.path.isfile(filename.format(counter)):
                            counter += 1

                        if counter > config.max_pic_save:
                            if os.path.exists(filename.format(1)):
                                os.remove(filename.format(1))
                                counter = 0
                        else:
                            if os.path.exists(filename.format(counter + 1)):
                                os.remove(filename.format(counter + 1))

                        filename = filename.format(counter)

                        norm_image = cv2.cvtColor(norm_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(filename, norm_image)


        pbar_whole = tqdm(total=(config.epochs * Io_tool.number_of_images), desc='Step')

        D_train_op = [network.D_optim, network.all_summary, network.summaries_dictionary['all/D_tot_GAN_loss']['list']]
        G_train_op = [network.G_optim, network.all_summary, network.summaries_dictionary['256x256/G_out_256']['list'] ,network.summaries_dictionary['all/D_tot_GAN_loss']['list']]

        D_Loss_min = 1000000
        G_Loss_min = 1000000
        no_G_impro = 0
        no_D_impro = 0
        step_type = 'Discriminator'
        D_step_count = 0
        while step < config.epochs * Io_tool.number_of_images:

            # if D_step_count < config.D_step:
            #     step_type = 'Discriminator'
            # else:
            #     D_step_count = 0
            #     step_type = 'Generator'

            ready_batch = Io_tool.compute_batch(Devices=len(available_gpus), Train_type = step_type)
            for batch in ready_batch:

                if step_type == 'Generator':
                    placeholder_dict = {network.z: batch['Z'],
                                        network.y_label: batch['Y_label'],
                                        network.x: batch['X'],
                                        network.y_fill: batch['Y_fill']}
                    t_op, summary, G_out, G_loss_list= sess.run(G_train_op, feed_dict=placeholder_dict)
                    G_loss = sum(G_loss_list)
                    step_type = 'Discriminator'
                    # if G_loss < G_Loss_min:
                    #     G_Loss_min = G_loss
                    #     no_G_impro = 0
                    # else:
                    #     no_G_impro += 1
                    #     if no_G_impro > config.max_no_G_impro:
                    #         G_Loss_min = 1000000
                    #         no_G_impro = 0
                    #         step_type = 'Discriminator'
                    #         break
                    log_writer.add_summary(summary, step)
                elif step_type == 'Discriminator':
                    placeholder_dict = {network.z: batch['Z'],
                                        network.y_label: batch['Y_label'],
                                        network.x: batch['X'],
                                        network.y_fill: batch['Y_fill']}
                    t_op, summary, D_loss_list= sess.run(D_train_op, feed_dict=placeholder_dict)
                    D_loss = sum(D_loss_list)
                    step_type = 'Generator'
                    # if config.D_step < D_step_count:
                    #     D_Loss_min = 1000000
                    #     no_D_impro = 0
                    #     step_type = 'Generator'
                    #     D_step_count = 0
                    #     break
                    # if D_loss < D_Loss_min:
                    #     D_Loss_min = D_loss
                    #     no_D_impro = 0
                    # else:
                    #     no_D_impro += 1
                    #     if no_D_impro > config.max_no_D_impro or D_loss<config.D_loss_thresh:
                    #         D_Loss_min = 1000000
                    #         no_D_impro = 0
                    #         step_type = 'Generator'
                    #         D_step_count = 0
                    #         break
                    log_writer.add_summary(summary, step)
                    D_step_count += 1
                    
                if step % config.save_step==0 and step !=0:
                    network.model_saver.save(sess, config.model_filename, global_step=step)
                    show_result()
                    
                step = step + config.batch_size*len(available_gpus)
                pbar_whole.update(config.batch_size*len(available_gpus))

        # images = []
        # for e in range(train_epoch):
        #     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        #     images.append(imageio.imread(img_name))
        # imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

        

train()