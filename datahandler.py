# import SimpleITK as sitk
from datashape.coretypes import float32
import directory_settings as s
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
import lmdb
import cv2
import scipy.io as sio

class DataHandler:
    def __init__(self):
        print('data handler')
        self.train_iterator = 0
        self.valid_iterator = 0
        self.view_iterator = 0

    def preprocess(self, data_batch, label_batch, rotate_degree=None, translate_std_ratio=None,
                   crop_width=None, crop_height=None, resize_width=None, resize_height=None,
                   normalize_to_1_scale=False, add_unit_channel=False):
        # todo: test every step by imshow
        # cv2.imshow('original', data_batch[0])
        if (resize_height, resize_width) is not (None, None):
            data_batch, label_batch = self.resize(data_batch, label_batch, resize_width, resize_height)
        if rotate_degree is not None:
            data_batch, label_batch = self.rotate_random(data_batch, label_batch, rotate_degree)
        if self.subtract_mean is True:
            data_batch = self.subtract_mean_image(data_batch)
        if translate_std_ratio is not None:
            self.translate_random(data_batch, label_batch, translate_std_ratio)
        if (crop_height, crop_width) is not (None, None):
            # if translate_std_ratio is not None:
            #     data_batch, label_batch = self.crop_random(data_batch, label_batch, crop_width, crop_height, translate_std_ratio)
            # else:
            data_batch, label_batch = self.crop_middle(data_batch, label_batch, crop_width, crop_height)
        if normalize_to_1_scale:
            data_batch = [img / 255. for img in data_batch]
        # return self.convert_to_npArray(data_batch, label_batch, scale_to_1=normalize_to_1_scale)
        if add_unit_channel:
            data_batch = self.add_unit_channel(data_batch)
        return (data_batch, label_batch)

    def add_unit_channel(self, imgs):
        imgs = np.asarray(imgs)
        return imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])

    def _read_train_valid_from_list_file(self, train_list, valid_list):
        load_to_memory = self.meta_data['load_to_memory']
        split_ratio = self.meta_data['split_ratio']
        train_images = train_labels = valid_images = valid_labels = []
        if train_list is not None and valid_list is not None:
            train_images, train_labels = self.read_data_from_list_file(train_list)
            valid_images, valid_labels = self.read_data_from_list_file(valid_list)
        elif train_list is not None and split_ratio != 0:
            images, labels = self.read_data_from_list_file(train_list)
            train_images, valid_images, train_labels, valid_labels = \
                train_test_split(images, labels, test_size=split_ratio)
        elif train_list is not None:
            train_images, train_labels = self.read_data_from_list_file(train_list)
            print('No validation set is defined, and no split ratio is set.')
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
            print('No validation set is defined, and no split ratio is set.')
            valid_labels, valid_images = [[], []]
        return train_images, train_labels, valid_images, valid_labels

    def _set_data_list_file(self, train_list_file, valid_list_file):
        self.valid_images = []
        self.train_images = []
        self.valid_labels = []
        self.train_labels = []
        for i in range(len(train_list_file)):
            train_images, train_labels, \
                valid_images, valid_labels = \
                self._read_train_valid_from_list_file(train_list_file[i], valid_list_file[i])
            self.train_images.append(train_images)
            self.train_labels.append(train_labels)
            self.valid_images.append(valid_images)
            self.valid_labels.append(valid_labels)

    def _set_data(self, data):
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
            data[0], data[1], data[2], data[3]

    def _set_data_folder(self, train_folder, valid_folder):
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
                                                self._read_train_valid_from_folder(train_folder, valid_folder,
                                                self.meta_data['file_format'], self.meta_data['split_ratio'])

    def set_data(self, data, meta_data):
        self.set_meta_data(meta_data)
        load_to_memory = meta_data['load_to_memory']
        label_type = meta_data['label_type']

        #set data
        if load_to_memory:
            print('loading all data into memory: %r')

        if label_type == 'single_value':
            if meta_data['train_list'] is not None:
                self._set_data_list_file(meta_data['train_list'], meta_data['valid_list'])
            elif data is not None:
                self._set_data(data)
        elif label_type == 'mask_image':
            # reading data from folder
            if meta_data['train_folder'] is not None:
                self._set_data_folder(meta_data['train_folder'], meta_data['valid_folder'])

        #update meta data
        self.update_meta_data({'training_size': self.get_dataset_size()[0],
                                            'valid_size': self.get_dataset_size()[1]})

        if meta_data['subtract_mean']:
            print('calculating mean image...')
            self.mean_train_image = self.set_train_mean()
            self.subtract_mean = True
        else:
            self.subtract_mean = False

        if 'main_label_index' in self.meta_data:
            main_label_index = self.meta_data['main_label_index']
        else:
            main_label_index = 0
        self.train_label_map, self.train_label_headers = self.create_label_map(self.train_labels, main_label_index)
        self.valid_label_map, self.valid_label_headers = self.create_label_map(self.valid_labels, main_label_index)

    def _read_data_from_folder(self, train_folder, image_format):
        print('_read_data_from_folder not implemented yet')
        return None, None

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_meta_data_json(self, meta_data):
        self.meta_data = meta_data

    def update_meta_data(self, input_meta_data):
        self.meta_data.update(input_meta_data)

    def get_meta_data(self):
        return self.meta_data

    def set_train_mean(self):
        load_to_memory = self.meta_data['load_to_memory']
        images = self.train_images;
        n = len(images)
        counter = 0
        if load_to_memory:
            h, w = images[0].size
        else:
            img = Image.open(images[0])
            h, w = img.size
        mean_im = np.zeros((w, h), np.float)
        for im in images:
            counter+=1
            if counter%100==0:
                print('read %d/%d images to calculate mean image' % (counter, n))
            if load_to_memory:
                imarr = np.array(im, dtype=float)/n
            else:
                imarr = np.array(Image.open(im), dtype=float)/n
            mean_im = mean_im + imarr
        return mean_im

    def subtract_mean_image(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = Image.fromarray(np.array(imgs[i], dtype=float) - self.mean_train_image)
        return imgs

    def read_data_from_list_file(self, list_file):
        delimiter = self.meta_data['delimiter']
        load_to_memory = self.meta_data['load_to_memory']
        file_format = self.meta_data['file_format']
        images = []
        labels = []
        counter = 0
        num_lines = sum(1 for line in open(list_file))


        # init_memory = self.memory_usage_ps()
        print('opening list file: {0}'.format(list_file))
        with open(list_file) as f:
            for line in f:
                counter+=1
                if counter%100==0:
                    print("read %d/%d lines from file: %s" % (counter, num_lines, list_file))
                str = line.rstrip().split(delimiter)
                file_dir = str[0]
                print(str)
                label = list(map(float, str[1::]))
                if os.path.isfile(file_dir):
                    if load_to_memory:
                        if file_format == 'mat':
                            matfile = sio.loadmat(file_dir)
                            dicomImage = matfile['Patient']['DicomImage']
                            images.append(dicomImage)
                        else:
                            images.append(Image.open(file_dir))
                            images[-1].load()
                    else:
                        images.append(file_dir)
                    labels.append(label)
                elif os.path.isdir(file_dir):
                    for sub_file in glob.iglob(file_dir+'/*'+file_format):
                        images.append(Image.open(sub_file) if load_to_memory else sub_file)  # .resize((400, 266), Image.BILINEAR)
                        if load_to_memory:
                            images[-1].load()
                        # img_arrray = cv2.imread(sub_file, 0)

                        # img_arrray = img_arrray.dtype(np.uint8)
                        # images.append(cv2.imread(sub_file,0 ) if flag else sub_file)
                        labels.append(int(label))
                        # print len(images), '  ', self.memory_usage_ps()-init_memory
                else:
                    print('file ', file_dir, ' does not exist')
                    assert False
        return images, labels

    def create_label_map(self, labels, main_label_index=0):
        number_of_views = len(labels)
        labels_map = []
        label_headers = []
        for view in range(number_of_views):
            sub_labels = labels[view]
            sub_main_labels = []
            for t in range(len(sub_labels)):
                sub_main_labels.append(sub_labels[t][main_label_index])
            sub_label_headers = sorted(set(sub_main_labels))
            number_of_classes = len(sub_label_headers)
            sub_labels_map = [[] for i in range(number_of_classes)]
            for i in range(len(sub_main_labels)):
                sub_labels_map[sub_label_headers.index(sub_main_labels[i])].append(i)
            labels_map.append(sub_labels_map)
            label_headers.append(sub_label_headers)
        return labels_map, label_headers

    def get_dataset_size(self):
        train_size = sum(len(l) for l in self.train_labels)
        valid_size = sum(len(l) for l in self.valid_labels)
        return train_size, valid_size

    def get_data_batch_random(self, batch_size, train_valid='train', method='uniform', view=None):
        main_label_index = self.meta_data['main_label_index']
        load_to_memory = self.meta_data['load_to_memory']
        label_type = self.meta_data['label_type']
        if train_valid == 'train':
            images = self.train_images[view]
            labels = self.train_labels[view]
            label_map = self.train_label_map[view]
        else:  # if train_valid == 'valid':
            images = self.valid_images[view]
            labels = self.valid_labels[view]
            label_map = self.valid_label_map[view]

        if method == 'random':
            selected_indices = np.random.permutation(len(images))[:batch_size]
        else:  # if method == 'uniform':
            num_classes = len(label_map)
            samples_per_class = batch_size//num_classes
            samples_per_class = 1 if samples_per_class == 0 else samples_per_class
            selected_indices = []
            for i in range(num_classes):
                indices = np.random.permutation(len(label_map[i]))[:samples_per_class]
                selected_indices.extend([label_map[i][j] for j in indices])
            if batch_size%num_classes != 0:
                indices = np.random.permutation(len(images))[:batch_size%num_classes]
                selected_indices.extend(indices)

        batch_images = []
        batch_labels = []
        if load_to_memory:
            batch_images = [images[i] for i in selected_indices]
            batch_labels = [labels[i] for i in selected_indices]
        else:
            # images = [Image.open(src_images[i]).load() for i in selected_indices]
            for i in selected_indices:
                if self.meta_data['file_format'] == 'image':
                    batch_images.append(Image.open(images[i]))
                    batch_images[-1].load()
                elif self.meta_data['file_format'] == 'mat':
                    matfile = sio.loadmat(images[i])
                    cine = matfile['Patient']['DicomImage'][0][0]  # todo: generalize this
                    batch_images.append(cine[:, :, 10])  # todo: read entire cine
            if label_type == 'single_value':
                batch_labels = [labels[i][main_label_index] for i in selected_indices]
            elif label_type == 'mask_image':
                # labels = [Image.open(src_labels[i]) for i in selected_indices]
                for i in selected_indices:
                    batch_labels.append(Image.open(labels[i]))
                    batch_labels[-1].load()
        return batch_images, batch_labels

    def get_data_batch_iterative(self, batch_size, train_valid='train'):
        load_to_memory = self.meta_data['load_to_memory']
        label_type = self.meta_data['label_type']
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
        if load_to_memory:
            images = [src_images[i] for i in selected_indices]
            labels = [src_labels[i] for i in selected_indices]
        else:
            images = [Image.open(src_images[i]) for i in selected_indices]
            if label_type == 'single_value':
                labels = [src_labels[i] for i in selected_indices]
            elif label_type == 'mask_image':
                labels = [Image.open(src_labels[i]) for i in selected_indices]

        if train_valid == 'train':
            self.train_iterator = iter
        elif train_valid == 'valid':
            self.valid_iterator = iter
        return images, labels

    def translate_random(self, imgs, labels, std_ratio=20):
        label_type = self.meta_data['label_type']
        origh, origw = imgs[0].shape

        for i in range(len(imgs)):
            transX = np.random.normal(origw / 2, origw/std_ratio)
            transY = np.random.normal(origh / 2, origh/std_ratio)
            if np.abs(transX) > 2*origw/std_ratio:
                transX = np.sign(transX)*2*origw/std_ratio
            if np.abs(transY) > 2 * origh / std_ratio:
                transY = np.sign(transY) * 2 * origh / std_ratio

            M = np.float32([[1, 0, transX], [0, 1, transY]])
            imgs[i] = dst = cv2.warpAffine(imgs[i], M, (origw, origh))

        if label_type == 'mask_image':
            raise NotImplementedError
            # todo: implement this

    def crop_random(self, imgs, labels, crop_width, crop_height, std_ratio=20):
        label_type = self.meta_data['label_type']

        #todo: add true random translation, not just cropping a random window because this requires padded images
        origh, origw = imgs[0].shape
        if [origw, origh] == [crop_width, crop_height]:
            return imgs, labels
        elif origw < crop_width or origh < crop_height:
            print('crop size is larger than the original (resized) image size. crop_random was interrupted.')
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
        label_type = self.meta_data['label_type']
        origh, origw = imgs[0].shape
        middlew = origw//2
        middleh = origh//2

        imgs = [
            im[middlew - crop_width // 2:middlew + crop_width // 2,
            middleh - crop_height // 2:middleh + crop_height // 2]
            for im in imgs]

        # for i in range(len(imgs)):
        #     imgs[i] = self.crop(imgs[i], middlew, middleh, crop_width, crop_height)

        if label_type == 'mask_image':
            raise NotImplementedError
            # labels[i] = self.crop(labels[i], middlew, middleh, crop_width, crop_height)

        return imgs, labels

    def crop(self, img, middlew, middleh, crop_width, crop_height):
        return img.crop((np.round(middlew - crop_width / 2. + 0.1).astype(int),
                                np.round(middleh - crop_height / 2. + 0.1).astype(int),
                                np.round(middlew + crop_width / 2. + 0.1).astype(int),
                                np.round(middleh + crop_height / 2. + 0.1).astype(int)))

    def rotate_random(self, imgs, labels, rotation_std):
        label_type = self.meta_data['label_type']
        origh, origw = imgs[0].shape

        #todo: added this during run. make sure it works!
        for i in range(len(imgs)):
            rot_degree = np.round(np.random.normal(0, rotation_std))
            # capping rotation to 2*std
            if np.abs(rot_degree) > 2*rotation_std:
               rot_degree = np.sign(rot_degree)*2*rotation_std

            M = cv2.getRotationMatrix2D((origw / 2, origh / 2), rot_degree, 1)
            imgs[i] = cv2.warpAffine(imgs[i] , M, (origw, origh))
            # imgs[i] = imgs[i].rotate(rot_degree)

            # todo: implement rotation for mask image
            # if label_type == 'mask_image':
            #     labels[i] = labels[i].rotate(rot_degree)
        return imgs, labels

    def convert_to_npArray(self, input_imgs, input_labels, scale_to_1=False):
        label_type = self.meta_data['label_type']
        height, width = input_imgs[0].size
        batch_size = len(input_imgs)
        image_array = np.ndarray((batch_size, 1, width, height), dtype=np.float32)
        if label_type == 'single_value':
            label_array = np.ndarray(batch_size, dtype=np.int8)
        elif label_type == 'mask_image':
            label_array = np.ndarray((batch_size, 1, width, height), dtype=np.int8)

        for i in range(batch_size):
            image_array[i, 0, :, :] = np.array(input_imgs[i], dtype=np.float32)
            if scale_to_1:
                image_array /= 255.
            if label_type == 'single_value':
                label_array[i] = input_labels[i]
            elif label_type == 'mask_image':
                label_array[i, 0, :, :] = np.array(input_labels[i], dtype=np.int8)
        return image_array, label_array

    def resize(self, imgs, labels, width, height):
        label_type = self.meta_data['label_type']
        origh, origw = imgs[0].shape
        if [origw, origh] == [width, height]:
            return imgs, labels

        imgs = [cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC) for image in imgs]
        # imgs = [image.resize((width, height), Image.BILINEAR) for image in imgs]

        # todo: implemnt image_mask resize
        # if label_type == 'image_mask':
        #     labels = [label.resize((width, height), Image.BILINEAR) for label in labels]
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