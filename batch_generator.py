import os
import cv2
import numpy as np
import random
import pprint
import time
import pickle
import datetime
import multiprocessing.dummy as mp
from tqdm import tqdm
from PIL import Image
import tools as tools
import config
import tools
from matplotlib import pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=4)


class IO_manager():
    def __init__(self):
        if (os.path.isfile('./dataset/' + config.action +'/dataset.pkl') and 
            os.path.isfile('./dataset/' + config.action +'/label_to_id.pkl') and
            os.path.isfile('./dataset/' + config.action +'/id_to_label.pkl') and
            os.path.isfile('./dataset/' + config.action +'/number_of_images.pkl') and
            os.path.isfile('./dataset/' + config.action +'/std.pkl') and
            os.path.isfile('./dataset/' + config.action +'/std_inverse.pkl') and
            os.path.isfile('./dataset/' + config.action +'/mean.pkl')
            and config.reuse_dataset):
            
            self.dataset = tools.load('dataset', 'dataset/' + config.action)
            self.label_to_id = tools.load('label_to_id', 'dataset/' + config.action)
            self.id_to_label = tools.load('id_to_label', 'dataset/' + config.action)
            self.number_of_images = tools.load('number_of_images', 'dataset/' + config.action)
            self.mean = tools.load('mean', 'dataset/' + config.action)
            self.std = tools.load('std', 'dataset/' + config.action)
            self.std_inverse = tools.load('std_inverse', 'dataset/' + config.action)
            self.number_of_classes = len(self.label_to_id)
        else:
            self.dataset, self.label_to_id, self.id_to_label, self.number_of_images = self.calculate_dataset()
            self.number_of_classes = len(self.label_to_id)
            self.mean = self.calculate_mean()
            self.std, self.std_inverse = self.calculate_std()
            tools.save(self.dataset, 'dataset', 'dataset/' + config.action)
            tools.save(self.label_to_id, 'label_to_id', 'dataset/' + config.action)
            tools.save(self.id_to_label, 'id_to_label', 'dataset/' + config.action)
            tools.save(self.number_of_images, 'number_of_images', 'dataset/' + config.action)
            tools.save(self.mean, 'mean', 'dataset/' + config.action)
            tools.save(self.std, 'std', 'dataset/' + config.action)
            tools.save(self.std_inverse, 'std_inverse', 'dataset/' + config.action)

        pp.pprint(self.std_inverse.max())
        pp.pprint(self.std_inverse.min())
        pp.pprint(self.mean.max())
        pp.pprint(self.mean.min())


    def calculate_dataset(self):
        dataset = {}
        label_to_id = {}
        id_to_label = {}
        number_of_images = 0
        index= -1
        for root, dirs, files in os.walk(config.dataset_folder):
            for fl in files:
                path = root + '/' + fl
                file_extension = fl.split('.')[-1]
                if file_extension == 'jpg':
                    folder_structure = root.split('/')
                    print(folder_structure)
                    # if folder_structure[-1] == 'ClassesIcons':
                    activity = folder_structure[-1]
                    if activity not in dataset:
                        dataset[activity] = []
                    n_images_for_current_activity = len(dataset[activity])
                    # if n_images_for_current_activity > config.max_images_for_activity:
                        # continue
                    if path not in dataset[activity]:
                        dataset[activity].append(path)
                        number_of_images += 1
                    if activity not in label_to_id:
                        index +=1
                        label_to_id[activity] = index
                        id_to_label[index] = activity
        
        return dataset, label_to_id, id_to_label, number_of_images

    def calculate_mean(self):
        n = 0
        mean = np.zeros([self.number_of_classes, config.H_size, config.W_size, 3])
        pbar = tqdm(total=(self.number_of_images), leave = False, desc='')
        for activity in self.dataset:
            a = self.label_to_id[activity]
            pbar.set_description('Computing mean of ' + activity)
            mean[a, ] = 0
            for path in self.dataset[activity]:
                # if n > 10:
                    # break
                img = cv2.imread(path).astype(np.float32)
                img = cv2.resize(img, dsize=(config.H_size, config.W_size), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_to_show = img.astype(np.uint8)
                value_mean = np.mean(img_to_show)
                if value_mean > 5:
                    if value_mean >= 5 and value_mean < 15:
                        img = self.improve_bright(img_to_show, clip_hist_percent=1).astype(np.float32)
                    norm_image = cv2.normalize(img, None, alpha=config.norm_min, beta=config.norm_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    mean[a, ...] = mean[a, ...] + norm_image
                pbar.update(1)
                n += 1
            mean[a, ...] = (mean[a, ...]/n)
            denorm_image = cv2.normalize(mean[a, ...], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            plt.imshow(denorm_image.astype(np.uint8))
            plt.show()
            n = 0
        return mean

    def calculate_std(self):
        n = 0
        std = {}
        std_inverse = {}
        pbar = tqdm(total=(self.number_of_images), leave = False, desc='')
        std = np.zeros([self.number_of_classes, config.mean_size, config.mean_size, 3])
        std_inverse = np.zeros([self.number_of_classes, config.mean_size, config.mean_size, 3])
        for activity in self.dataset:
            a = self.label_to_id[activity]
            pbar.set_description('Computing standard deviation of ' + activity)
            mean = self.mean[a, ...]
            mean = cv2.resize(mean, dsize=(config.mean_size, config.mean_size), interpolation=cv2.INTER_CUBIC)

            for path in self.dataset[activity]:
                img = cv2.imread(path)
                img = cv2.resize(img, dsize=(config.mean_size, config.mean_size), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_to_show = img.astype(np.uint8)
                value_mean = np.mean(img_to_show)
                if value_mean > 5:
                    if value_mean >= 5 and value_mean < 15:
                        img = self.improve_bright(img_to_show, clip_hist_percent=1).astype(np.float32)
                    norm_image = cv2.normalize(img, None, alpha=config.norm_min, beta=config.norm_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    diff = img.astype(np.float32) - mean.astype(np.float32)
                    diff_t = np.transpose(diff, axes=[1, 0, 2])
                    for j in range(3):
                        std[a, :, :, j] += np.matmul(diff[:, :, j], diff_t[:, :, j])
                    n += 1
                pbar.update(1)
            std[a, ...] = (std[a, ...]/n).astype(np.float32)
            denorm_image = cv2.normalize(std[a, ...], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            plt.imshow(denorm_image.astype(np.uint8))
            plt.show()
            pbar.set_description('Computing Inverse of' + activity)
            pbar.update(0)
            for j in range(3):
                channel = std[a, :, :, j]
                # c = np.linalg.inv(np.linalg.cholesky(channel))
                # inverse = np.dot(c.T,c)
                inverse = np.linalg.inv(channel)
                std_inverse[a, :, :, j] = inverse
            denorm_image = cv2.normalize(std_inverse[a, ...], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            plt.imshow(denorm_image.astype(np.uint8))
            plt.show()
            n = 0
        return std, std_inverse

    
    def compute_batch(self, Devices, Train_type):

        def multiprocess_batch(x):
            X, Y_label, Y_fill, Z = self.batch_generator(pbar, 'Discriminator', Devices)
            return {'X': X,
                    'Y_label': Y_label,
                    'Y_fill': Y_fill,
                    'Z': Z}
        

        if Train_type == 'Discriminator':
            desc = 'Training_Discriminator'
            pbar = tqdm(total=(config.batch_size*Devices*config.tasks), leave = False, desc=desc)
        else:
            desc = 'Training_Generator'
            pbar = tqdm(total=(config.batch_size*Devices*config.tasks), leave = False, desc=desc)

        pool = mp.Pool(processes=config.processes)
        generated_batch = pool.map(multiprocess_batch, range(0, config.tasks))
        ready_batch = generated_batch.copy()
        pbar.close()
        pool.close()
        pool.join()
        return ready_batch

    def improve_bright(self, image, clip_hist_percent=1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        blur = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        new_image = cv2.bilateralFilter(blur.astype(np.float32),11,95,95)
        return new_image

    def batch_generator(self, pbar, Train_type, Devices):
        random.seed(time.time())

        z = np.random.normal(0, 1, (Devices, config.batch_size, config.noise_size))
        y = np.random.randint(0, self.number_of_classes, (Devices, config.batch_size, 1))
        y_label = np.eye(self.number_of_classes + 1)[y.astype(np.int32)].reshape([Devices, config.batch_size, 1, 1, self.number_of_classes + 1])
        y_fill = y_label * np.ones([Devices, config.batch_size, config.num_patches, config.num_patches, self.number_of_classes + 1])
        X = np.ones(shape=(Devices, config.batch_size, config.H_size, config.W_size, config.input_channels), dtype=np.float32)

        if Train_type == 'Discriminator':
            # y_fill = np.ones([Devices, config.batch_size, config.H_size, config.W_size, self.number_of_classes])

            d = 0
            while d < Devices:
                j = 0
                while j < config.batch_size:
                    random_activity_id = np.argmax(y_fill[d, j, 1, 1, :])
                    random_activity = self.id_to_label[random_activity_id]
                    # random_activity = random.choice(list(self.dataset.keys()))
                    random_iage_path = random.choice(self.dataset[random_activity])

                    try:
                        img = cv2.imread(random_iage_path)
                        img_to_show = img.astype(np.uint8)
                        value_sum = sum(sum(sum(img_to_show)))
                        value_mean = np.mean(img_to_show)
                        if value_mean > 5:
                            if value_mean >= 5 and value_mean < 15:
                                img = self.improve_bright(img_to_show, clip_hist_percent=1)

                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if img.shape[0] != config.H_size and img.shape[0] != config.W_size:
                                img = cv2.resize(img, dsize=(config.H_size, config.W_size), interpolation=cv2.INTER_CUBIC)

                            norm_image = cv2.normalize(img, None, alpha=config.norm_min, beta=config.norm_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                            X[d, j, :, :, :] = norm_image
                            j = j + 1
                    except Exception as e:
                        print(e)
                        print(random_iage_path)


                    # modified = y_fill[d, j, :, :, :] * self.label_to_id[random_activity]
                    # y_fill[d, j, :, :, :] = modified

                    pbar.update(1)
                    
                d = d + 1

        pbar.update(1)
        pbar.refresh()
        return X, y_label, y_fill, z
