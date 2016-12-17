# import SimpleITK as sitk
from datashape.coretypes import float32
import directory_settings as s
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cross_validation import train_test_split
import lmdb
#import cv2

class data_handler:
    def __init__(self, label_type, load_to_memory):
        print 'data handler'
        self.train_iterator = 0
        self.valid_iterator = 0
        self.label_type = label_type
        self.load_to_memory = load_to_memory

    def preprocess(self, data_batch, label_batch, rotate_degree=None, translate_std_ratio=None,
                   crop_width=None, crop_height=None, resize_width=None, resize_height=None, normalize_to_1_scale=False):
        if (resize_height, resize_width) is not (None, None):
            data_batch, label_batch = self.resize(data_batch, label_batch, (resize_width, resize_height))
        if rotate_degree is not None:
            data_batch, label_batch = self.rotate_random(data_batch, label_batch, rotate_degree)
        if self.subtract_mean is True:
            data_batch = self.subtract_mean_image(data_batch)
        if (crop_height, crop_width) is not (None, None):
            if translate_std_ratio is not None:
                data_batch, label_batch = self.crop_random(data_batch, label_batch, crop_width, crop_height, translate_std_ratio)
            else:
                data_batch, label_batch = self.crop_middle(data_batch, label_batch, crop_width, crop_height)

        return self.convert_to_npArray(data_batch, label_batch, scale_to_1=normalize_to_1_scale)

    def _read_train_valid_from_list_file(self, train_list, valid_list, image_format, split_ratio):
        train_images = train_labels = valid_images = valid_labels = []
        if train_list is not None and valid_list is not None:
            train_images, train_labels = self.read_data_from_list_file(train_list, image_format, self.load_to_memory)
            valid_images, valid_labels = self.read_data_from_list_file(valid_list, image_format, self.load_to_memory)
        elif train_list is not None and split_ratio != 0:
            images, labels = self.read_data_from_list_file(train_list, image_format, self.load_to_memory)
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(images, labels, test_size=split_ratio)
        elif train_list is not None:
            train_images, train_labels = self.read_data_from_list_file(train_list, image_format, self.load_to_memory)
            print 'No validation set is defined, and no split ratio is set.'
            [valid_labels, valid_images] = [[], []]
        return train_images, train_labels, valid_images, valid_labels

    def _read_train_valid_from_folder(self, train_folder, valid_folder, image_format, split_ratio):
        train_images = train_labels = valid_images = valid_labels = []
        if train_folder is not None and valid_folder is not None:
            train_images, train_labels = self._read_data_from_folder(train_folder, image_format)
            valid_images, valid_labels = self._read_data_from_folder(valid_folder, image_format)
        elif train_folder is not None and split_ratio != 0:
            images, labels = self._read_data_from_folder(train_folder, image_format)
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(images, labels, test_size=split_ratio)
        elif train_folder is not None:
            train_images, train_labels = self._read_data_from_folder(train_folder, image_format)
            print 'No validation set is defined, and no split ratio is set.'
            valid_labels, valid_images = [[], []]
        return train_images, train_labels, valid_images, valid_labels

    def _set_data_list_file(self, train_list_file, valid_list_file):
        self.train_images, self.train_labels, \
        self.valid_images, self.valid_labels = \
            self._read_train_valid_from_list_file(train_list_file, valid_list_file,
                                                  self.meta_data['image_format'], self.meta_data['split_ratio'])

    def _set_data(self, data):
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
            data[0], data[1], data[2], data[3]

    def _set_data_folder(self, train_folder, valid_folder):
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
                                                self._read_train_valid_from_folder(train_folder, valid_folder,
                                                self.meta_data['image_format'], self.meta_data['split_ratio'])
        # augmenter = DataAugmentor(train_folder=train_folder, validation_folder=None,
        #                           augment=False, split_ratio=0.2, type='segmentation', same_folder=True)
        # self.train_images, self.valid_images, self.train_labels, self.valid_labels = augmenter.get_all_data()

    def set_data(self, train_list_file=None, valid_list_file=None,
                 train_folder=None, valid_folder=None,
                 data=None,
                 image_format='.jpg', split_ratio=0, load_to_memory=False, subtract_mean=True):

        print 'setting data...'
        self.set_meta_data(train_list=train_list_file, valid_list=valid_list_file, train_folder=train_folder,
                           valid_folder=valid_folder,
                           image_format=image_format, split_ratio=split_ratio,
                           load_to_memory=load_to_memory, subtract_mean=subtract_mean)
        #set data
        print 'loading all data into memory: %r' % load_to_memory
        if self.label_type == 'single_value':
            if train_list_file is not None:
                self._set_data_list_file(train_list_file, valid_list_file)
            elif data is not None:
                self._set_data(data)
        elif self.label_type == 'mask_image':
            # reading data from folder
            if train_folder is not None:
                self._set_data_folder(train_folder, valid_folder)

        #update meta data
        self.update_meta_data({'training_size': self.get_dataset_size()[0],
                                            'valid_size': self.get_dataset_size()[1]})

        if subtract_mean:
            print 'calculating mean image...'
            self.mean_train_image = self.set_train_mean()
            self.subtract_mean = True
        else:
            self.subtract_mean = False

        self.train_label_map = self.create_label_map(self.train_labels)
        self.valid_label_map = self.create_label_map(self.valid_labels)

    def _read_data_from_folder(self, train_folder, image_format):
        print '_read_data_from_folder not implemented yet'
        return None, None

    def set_meta_data(self, train_list=None, valid_list=None, image_format='.jpg', split_ratio=0, load_to_memory=False,
                 subtract_mean=True, train_folder=None, valid_folder=None):
        self.meta_data = {'train_list': train_list, 'train_folder': train_folder,
                               'valid_list': valid_list, 'valid_folder': valid_folder,
                               'split_ratio': split_ratio, 'load_to_memory': load_to_memory,
                               'subtract_mean': subtract_mean, 'image_format': image_format}

    def set_meta_data_json(self, meta_data):
        self.meta_data = meta_data

    def update_meta_data(self, input_meta_data):
        self.meta_data.update(input_meta_data)

    def get_meta_data(self):
        return self.meta_data

    def set_train_mean(self):
        images = self.train_images;
        n = len(images)
        counter = 0
        if self.load_to_memory:
            h, w = images[0].size
        else:
            img = Image.open(images[0])
            h, w = img.size
        mean_im = np.zeros((w, h), np.float)
        for im in images:
            counter+=1
            if counter%100==0:
                print 'read %d/%d images to calculate mean image' % (counter, n)
            if self.load_to_memory:
                imarr = np.array(im, dtype=float)/n
            else:
                imarr = np.array(Image.open(im), dtype=float)/n
            mean_im = mean_im + imarr
        return mean_im

    def subtract_mean_image(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = Image.fromarray(np.array(imgs[i], dtype=float) - self.mean_train_image)
        return imgs

    @staticmethod
    def read_data_from_list_file(list_file, image_format='.jpg', load_to_memory=False):
        images = []
        labels = []
        counter = 0
        num_lines = sum(1 for line in open(list_file))
        flag_memory = load_to_memory

        # init_memory = self.memory_usage_ps()
        with open(list_file) as f:
            for line in f:
                counter+=1
                if counter%100==0:
                    print "read %d/%d lines from file: %s" % (counter, num_lines, list_file)
                file_dir, label = line.split()
                if os.path.isfile(file_dir):
                    images.append(Image.open(file_dir) if flag_memory else file_dir)
                    if flag_memory:
                        images[-1].load()
                    labels.append(int(label))
                elif os.path.isdir(file_dir):
                    for sub_file in glob.iglob(file_dir+'/*'+image_format):
                        images.append(Image.open(sub_file) if flag_memory else sub_file)  # .resize((400, 266), Image.BILINEAR)
                        if flag_memory:
                            images[-1].load()
                        # img_arrray = cv2.imread(sub_file, 0)

                        # img_arrray = img_arrray.dtype(np.uint8)
                        # images.append(cv2.imread(sub_file,0 ) if flag else sub_file)
                        labels.append(int(label))
                        # print len(images), '  ', self.memory_usage_ps()-init_memory
                else:
                    print 'file ', file_dir, ' does not exist'
                    assert False
        return images, labels

    def create_label_map(self, labels):
        label_headerss = sorted(set(labels))
        labels_map = [[] for i in range(6)]
        for i in range(len(labels)):
            labels_map[labels[i]].append(i)
        return labels_map

    def get_dataset_size(self):
        return len(self.train_labels), len(self.valid_labels)

    def get_data_batch_random(self, batch_size, train_valid='train', method='uniform'):
        if train_valid == 'train':
            src_images = self.train_images
            src_labels = self.train_labels
            src_label_map = self.train_label_map
        elif train_valid == 'valid':
            src_images = self.valid_images
            src_labels = self.valid_labels
            src_label_map = self.valid_label_map

        if method == 'random':
            selected_indices = np.random.permutation(len(src_images))[:batch_size]
        elif method == 'uniform':
            num_classes = len(src_label_map)
            samples_per_class = round(batch_size/num_classes)
            selected_indices = []
            for i in range(num_classes):
                indices = np.random.permutation(len(src_label_map[i]))[:samples_per_class]
                selected_indices.extend([src_label_map[i][j] for j in indices])
            if batch_size%num_classes != 0:
                indices = np.random.permutation(len(src_images))[:batch_size%num_classes]
                selected_indices.extend(indices)

        images = []
        labels = []
        if self.load_to_memory:
            images = [src_images[i] for i in selected_indices]
            labels = [src_labels[i] for i in selected_indices]
        else:
            # images = [Image.open(src_images[i]).load() for i in selected_indices]
            for i in selected_indices:
                images.append(Image.open(src_images[i]))
                images[-1].load()
            if self.label_type == 'single_value':
                labels = [src_labels[i] for i in selected_indices]
            elif self.label_type == 'mask_image':
                # labels = [Image.open(src_labels[i]) for i in selected_indices]
                for i in selected_indices:
                    labels.append(Image.open(src_labels[i]))
                    labels[-1].load()
        return images, labels

    def get_data_batch_iterative(self, batch_size, train_valid='train'):
        if train_valid == 'train':
            src_images = self.train_images
            src_labels = self.train_labels
            iter = self.train_iterator
        elif train_valid == 'valid':
            src_images = self.valid_images
            src_labels = self.valid_labels
            iter = self.valid_iterator
        selected_indices = np.array(range(iter, iter+batch_size))
        selected_indices[selected_indices>=len(src_images)] = selected_indices[selected_indices>=len(src_images)]-len(src_images)
        iter = selected_indices[batch_size-1]+1
        if self.load_to_memory:
            images = [src_images[i] for i in selected_indices]
            labels = [src_labels[i] for i in selected_indices]
        else:
            images = [Image.open(src_images[i]) for i in selected_indices]
            if self.label_type == 'single_value':
                labels = [src_labels[i] for i in selected_indices]
            elif self.label_type == 'mask_image':
                labels = [Image.open(src_labels[i]) for i in selected_indices]

        if train_valid == 'train':
            self.train_iterator = iter
        elif train_valid == 'valid':
            self.valid_iterator = iter
        return images, labels

    def crop_random(self, imgs, labels, crop_width, crop_height, std_ratio=20):
        #todo: add true random translation, not just cropping a random window because this requires padded images
        origw, origh = imgs[0].size
        if [origw, origh] == [crop_width, crop_height]:
            return imgs, labels
        elif origw < crop_width or origh < crop_height:
            print 'crop size is larger than the original (resized) image size. crop_random was interrupted.'
            return imgs, labels
        for i in range(len(imgs)):
            middlew = np.random.normal(origw / 2, origw/std_ratio)
            middleh = np.random.normal(origh / 2, origh/std_ratio)
            if middlew+crop_width/2 > origw:
                middlew = origw - crop_width/2
            elif middlew-crop_width/2 < 0:
                middlew = crop_width/2
            if middleh+crop_height/2 > origh:
                middleh = origh - crop_height/2
            elif middleh-crop_height/2 < 0:
                middleh = crop_height/2
            imgs[i] = self.crop(imgs[i], middlew, middleh, crop_width, crop_height)
            assert imgs[i].size == (crop_width, crop_height)
            if self.label_type == 'mask_image':
                labels[i] = self.crop(labels[i], middlew, middleh, crop_width, crop_height)
        return imgs, labels

    def crop_middle(self, imgs, labels, crop_width, crop_height):
        origw, origh = imgs[0].size
        middlew = origw/2
        middleh = origh/2
        for i in range(len(imgs)):
            imgs[i] = self.crop(imgs[i], middlew, middleh, crop_width, crop_height)
            if self.label_type == 'mask_image':
                labels[i] = self.crop(labels[i], middlew, middleh, crop_width, crop_height)
        return imgs, labels

    def crop(self, img, middlew, middleh, crop_width, crop_height):
        return img.crop((np.round(middlew - crop_width / 2. + 0.1).astype(int),
                                np.round(middleh - crop_height / 2. + 0.1).astype(int),
                                np.round(middlew + crop_width / 2. + 0.1).astype(int),
                                np.round(middleh + crop_height / 2. + 0.1).astype(int)))

    def rotate_random(self, imgs, labels, rotation_std):
        #todo: added this during run. make sure it works!
        for i in range(len(imgs)):
            rot_degree = np.round(np.random.normal(0, rotation_std))
            if np.abs(rot_degree) > 2*rotation_std:
               rot_degree = np.sign(rot_degree)*2*rotation_std
            # if np.random.rand(1) > 0.5:
            #     rot_degree = - rot_degree
            imgs[i] = imgs[i].rotate(rot_degree)
            if self.label_type == 'mask_image':
                labels[i] = labels[i].rotate(rot_degree)
        return imgs, labels

    def convert_to_npArray(self, input_imgs, input_labels, scale_to_1=False):
        height, width = input_imgs[0].size
        batch_size = len(input_imgs)
        image_array = np.ndarray((batch_size, 1, width, height), dtype=np.float32)
        if self.label_type == 'single_value':
            label_array = np.ndarray((batch_size), dtype=np.int8)
        elif self.label_type == 'mask_image':
            label_array = np.ndarray((batch_size, 1, width, height), dtype=np.int8)

        for i in range(batch_size):
            image_array[i, 0, :, :] = np.array(input_imgs[i], dtype=np.float32)
            if scale_to_1:
                image_array /= 255.
            if self.label_type == 'single_value':
                label_array[i] = input_labels[i]
            elif self.label_type == 'mask_image':
                label_array[i, 0, :, :] = np.array(input_labels[i], dtype=np.int8)
        return image_array, label_array

    def resize(self, imgs, labels, (width, height)):
        origw, origh = imgs[0].size
        if [origw, origh] == [width, height]:
            return imgs, labels
        imgs = [image.resize((width, height), Image.BILINEAR) for image in imgs]
        # for image in imgs:
        #     image = image.resize((width, height), Image.BILINEAR)
        if self.label_type == 'image_mask':
            labels = [label.resize((width, height), Image.BILINEAR) for label in labels]
            # for label in labels:
            #     label = label.resize((width, height), Image.BILINEAR)
        return imgs, labels


# def memory_usage_resource(self):
    #     import resource
    #     import sys
    #     rusage_denom = 1024.
    #     if sys.platform == 'darwin':
    #         # ... it seems that in OSX the output is different units ...
    #         rusage_denom = rusage_denom * rusage_denom
    #     mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    #     return mem
    #
    # def memory_usage_ps(self):
    #     import subprocess
    #     out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
    #                            stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    #     vsz_index = out[0].split().index(b'RSS')
    #     mem = float(out[1].split()[vsz_index]) / 1024
    #     return mem

    #todo: add random scaling