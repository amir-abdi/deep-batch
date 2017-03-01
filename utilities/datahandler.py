import gc
import glob
import os
import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self):
        print('data handler')
        self.train_labels = None
        self.test_labels = None
        self.valid_labels = None

    def preprocess(self, data_batch, label_batch):
        meta_data = self.meta_data
        rotate_degree = meta_data['random_rotate_value']
        translate_std_ratio = meta_data['random_translate_ratio_value']
        crop_width = meta_data['crop_width']
        crop_height = meta_data['crop_height']
        resize_width = meta_data['resize_width']
        resize_height = meta_data['resize_height']
        scale_data_value = meta_data['scale_data']
        scale_label_value = meta_data['scale_label']
        stream_specific_scale_label = meta_data['stream_specific_scale_label']
        reshape_batch = meta_data['reshape_batch']

        # cv2.imshow('original', data_batch[0])
        if (resize_height, resize_width) is not (None, None):
            data_batch, label_batch = self.resize(data_batch, label_batch, resize_width, resize_height)
        if rotate_degree is not None:
            data_batch, label_batch = self.rotate_random(data_batch, label_batch, rotate_degree)
        if self.meta_data['subtract_mean'] is True:
            data_batch = self.subtract_mean_image(data_batch)
        if translate_std_ratio is not None:
            self.translate_random(data_batch, label_batch, translate_std_ratio)
        if (crop_height, crop_width) is not (None, None):
            data_batch, label_batch = self.crop_middle(data_batch, label_batch, crop_width, crop_height)

        if scale_data_value != 1:
            data_batch = self.scale_data(data_batch, scale_data_value)
        if stream_specific_scale_label is not None:
            label_batch = self.streamspecific_scale_label(label_batch, stream_specific_scale_label, self.current_view)

        data_batch = np.asarray(data_batch)
        if reshape_batch == 'tf':
            data_batch = self.match_TensorFlow_shape(data_batch)
        elif reshape_batch == 'caffe':
            data_batch = self.match_caffe_shape(data_batch)

        return (data_batch, label_batch)

    def scale_data(self, data_batch, scale_data_value):
        return [img / scale_data_value for img in data_batch]

    def streamspecific_scale_label(self, labels, stream_label_ranges, current_stream):
        temp = [l / stream_label_ranges[current_stream] for l in labels]
        # fix error in labels if above 1
        temp = [label if label <= 1 else 1 for label in temp]
        return temp

    def match_caffe_shape(self, imgs):
        num_seq = self.meta_data['sequence_length']
        channels = self.meta_data['channels']

        if self.meta_data['sequence_length'] != 1:
            raise NotImplementedError
        else:
            imgs = np.swapaxes(imgs, 1, 3)
            imgs = np.swapaxes(imgs, 2, 3)

            if channels == 1:
                imgs = imgs.reshape(imgs.shape[0],  # batch size
                                    channels,  # channels
                                    imgs.shape[1],  # width
                                    imgs.shape[2],  # height
                                    )
            else:
                # todo: test
                pass
        return imgs

    def match_TensorFlow_shape(self, imgs):
        # For 3D data, "tf" assumes (conv_dim1, conv_dim2, conv_dim3, channels)
        # while "th" assumes  (channels, conv_dim1, conv_dim2, conv_dim3).
        num_seq = self.meta_data['sequence_length']
        channels = self.meta_data['channels']
        if self.meta_data['sequence_length'] != 1:
            imgs = np.swapaxes(imgs, 1, 3)
            imgs = np.swapaxes(imgs, 2, 3)
            imgs = imgs.reshape(imgs.shape[0],  # batch size
                                imgs.shape[1],  # num_frames
                                imgs.shape[2],  # width
                                imgs.shape[3],  # height
                                channels        # channels
                                )
        else:
            imgs = imgs.reshape(imgs.shape[0],  # batch size
                                imgs.shape[1],  # width
                                imgs.shape[2],  # height
                                channels        # channels
                                )
        return imgs

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

    def set_test_data(self, meta_data):
        self.set_meta_data(meta_data)
        load_to_memory = meta_data['load_to_memory']
        label_type = meta_data['label_type']
        test_list_file  = meta_data['test_list_file']

        self.test_images = []
        self.test_labels = []
        for i in range(len(test_list_file)):
            test_images, test_labels = self.read_data_from_list_file(test_list_file[i])
            self.test_images.append(test_images)
            self.test_labels.append(test_labels)


        if 'main_label_index' in self.meta_data:
            main_label_index = self.meta_data['main_label_index']
        else:
            main_label_index = 0
        self.test_label_map, self.test_label_headers = self.create_label_map(self.test_labels, main_label_index)
        self.test_iterator = np.zeros(self.get_num_views(), np.int)

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

        if 'main_label_index' in self.meta_data:
            main_label_index = self.meta_data['main_label_index']
        else:
            main_label_index = 0
        self.train_label_map, self.train_label_headers = self.create_label_map(self.train_labels, main_label_index)
        self.valid_label_map, self.valid_label_headers = self.create_label_map(self.valid_labels, main_label_index)

        self.train_iterator = np.zeros(self.get_num_views(), np.int)
        self.valid_iterator = np.zeros(self.get_num_views(), np.int)

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
        multi_cine_per_patient = self.meta_data['multi_cine_per_patient']
        file_format = self.meta_data['file_format']
        num_frames = self.meta_data['sequence_length']
        images = []
        labels = []
        counter = 0
        num_lines = sum(1 for line in open(list_file))

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
                            cines = self.read_patient_from_mat(file_dir, multi_cine_per_patient=multi_cine_per_patient)
                            images.extend(cines)
                            for i in range(len(cines)):
                                labels.append(label)
                            # matfile = sio.loadmat(file_dir)
                            # cine = matfile['Patient']['DicomImage'][0][0]  # todo: generalize this
                            # images.append(cine[:, :, :num_frames])
                        else:
                            images.append(Image.open(file_dir))
                            images[-1].load()
                            labels.append(label)
                    else:
                        images.append(file_dir)
                        labels.append(label)
                elif os.path.isdir(file_dir):
                    for sub_file in glob.iglob(file_dir+'/*'+file_format):
                        images.append(Image.open(sub_file) if load_to_memory else sub_file)  # .resize((400, 266), Image.BILINEAR)
                        if file_format == 'image':
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
        # if self.meta_data['multi_stream']:
        number_of_streams = len(labels)
        # else:
        #     number_of_streams = 1
        labels_map = []
        label_headers = []
        for stream in range(number_of_streams):
            sub_labels = labels[stream]
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

    def get_testset_size(self):
        # test_size = sum(len(l) for l in self.test_labels)
        test_size = max(len(l) for l in self.test_labels)
        return test_size

    def get_testset_size_per_view(self):
        return [len(l) for l in self.test_labels]

    def get_num_views(self):
        if self.train_labels is not None:
            return len(self.train_labels)
        elif self.test_labels is not None:
            return len(self.test_labels)
        else:
            return 0

    def get_batch(self, batch_size, train_valid='train', data_traversing='random',
                  stream_index=0):
        interclass_selection_method = self.meta_data['interclass_batch_selection']
        self.current_view = stream_index
        main_label_index = self.meta_data['main_label_index']
        load_to_memory = self.meta_data['load_to_memory']
        label_type = self.meta_data['label_type']
        num_frames = self.meta_data['sequence_length']
        multi_stream_flag = self.meta_data['multi_stream']

        if train_valid == 'train':
            images = self.train_images[stream_index]
            labels = self.train_labels[stream_index]
            label_map = self.train_label_map[stream_index]
            iter = self.train_iterator[stream_index]
        elif train_valid == 'valid':
            images = self.valid_images[stream_index]
            labels = self.valid_labels[stream_index]
            label_map = self.valid_label_map[stream_index]
            iter = self.valid_iterator[stream_index]
        elif train_valid == 'test':
            images = self.test_images[stream_index]
            labels = self.test_labels[stream_index]
            label_map = self.test_label_map[stream_index]
            iter = self.test_iterator[stream_index]

        if data_traversing == 'random':
            if interclass_selection_method == 'random':
                selected_indices = np.random.permutation(len(images))[:batch_size]
            elif interclass_selection_method == 'uniform':
                num_classes = len(label_map)
                samples_per_class = batch_size//num_classes
                selected_indices = []
                for i in range(num_classes):
                    indices = np.random.permutation(len(label_map[i]))[:samples_per_class]
                    while len(indices) < samples_per_class:
                        indices = np.append(indices, np.random.permutation(len(label_map[i]))[:samples_per_class-len(indices)])
                    selected_indices.extend([label_map[i][j] for j in indices])

                if batch_size % num_classes != 0:
                    selected_classes = np.random.permutation(num_classes)[:batch_size%num_classes]
                    for i in range(len(selected_classes)):
                        index = np.random.randint(len(label_map[selected_classes[i]]))
                        selected_indices.extend([label_map[selected_classes[i]][index]])




            else:
                assert False
        elif data_traversing == 'iterative':
            selected_indices = np.array(range(iter, iter + batch_size))
            selected_indices[selected_indices >= len(images)] = \
                selected_indices[selected_indices >= len(images)] - len(images)
            iter = selected_indices[batch_size - 1] + 1  # todo: switch to view specific iter
            if train_valid == 'train':
                self.train_iterator[stream_index] = iter
            elif train_valid == 'valid':
                self.valid_iterator[stream_index] = iter
            elif train_valid == 'test':
                self.test_iterator[stream_index] = iter

        # print("view: {}".format(view), "  iter: {}".format(iter))

        batch_images = []
        batch_labels = []
        if load_to_memory:
            batch_images = [images[i] for i in selected_indices]
            batch_labels = [labels[i][main_label_index] for i in selected_indices]
        else:
            # images = [Image.open(src_images[i]).load() for i in selected_indices]
            for i in selected_indices:
                if self.meta_data['file_format'] == 'image':
                    im = cv2.imread(images[i], 0)
                    batch_images.append(im)
                    # batch_images.append(Image.open(images[i]))
                    # batch_images[-1].load()
                elif self.meta_data['file_format'] == 'mat':
                    cines = self.read_patient_from_mat(images[i], multi_cine_per_patient=False)
                    batch_images.extend(cines)  # todo: read entire cine
                    # print('Number of frames = ', str(cines[0].shape[2]))
            if label_type == 'single_value':
                batch_labels = [labels[i][main_label_index] for i in selected_indices]
            elif label_type == 'mask_image':
                # labels = [Image.open(src_labels[i]) for i in selected_indices]
                for i in selected_indices:
                    batch_labels.append(Image.open(labels[i]))
                    batch_labels[-1].load()
        return batch_images, batch_labels

    def read_patient_from_mat(self, file, multi_cine_per_patient=False):
        cine_selection_if_not_multi = self.meta_data['cine_selection_if_not_multi']
        num_frames = self.meta_data['sequence_length']
        matfile = sio.loadmat(file)
        cine = matfile['Patient']['DicomImage'][0][0]  # todo: generalize this
        cines = []
        if cine.shape[2] == num_frames:
            cines.append(np.copy(cine))
        elif cine.shape[2] > num_frames:
            if multi_cine_per_patient:
                # consider every num_frames frames as one patient
                i = 0
                while i+num_frames < cine.shape[2]:
                    temp_cine = cine[:, :, i:i+num_frames]
                    cines.append(np.copy(temp_cine))
                    i += num_frames
            else:
                if cine_selection_if_not_multi == 'random':  # choose one random sequence of num_frames length
                    from random import randint
                    i = randint(0, cine.shape[2] - num_frames)
                elif cine_selection_if_not_multi == 'first':  # choose the first cine sequence
                    i = 0
                cines.append(np.copy(cine[:, :, i:i + num_frames]))
        elif cine.shape[2] < num_frames:
            # cycle over
            # cine = np.resize(cine, (cine.shape[0], cine.shape[1], num_frames))
            cine = np.concatenate((cine, cine[:, :, :num_frames-cine.shape[2]]), axis=2)
            cines.append(np.copy(cine))
        gc.collect()
        return cines

    def translate_random(self, imgs, labels, value=20):
        label_type = self.meta_data['label_type']
        method = self.meta_data['random_translate_method']
        origh, origw, __ = imgs[0].shape

        for i in range(len(imgs)):
            if method == 'normal':
                transX = np.random.normal(0, origw / value)
                transY = np.random.normal(0, origh / value)
                if np.abs(transX) > 2*origw/value:
                    transX = np.sign(transX)*2*origw / value
                if np.abs(transY) > 2 * origh / value:
                    transY = np.sign(transY) * 2 * origh / value
            elif method == 'uniform':
                transX = np.random.uniform(-(origw / value),
                                           (origw / value))
                transY = np.random.uniform(- (origh / value),
                                           (origh / value))

            M = np.float32([[1, 0, transX], [0, 1, transY]])
            imgs[i] = dst = cv2.warpAffine(imgs[i], M, (origw, origh))

        if label_type == 'mask_image':
            raise NotImplementedError
            # todo: implement this

    def crop_random(self, imgs, labels, crop_width, crop_height, std_ratio=20):
        label_type = self.meta_data['label_type']
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
        origh, origw, __ = imgs[0].shape

        if [origw, origh] == [crop_width, crop_height]:
            return imgs, labels

        middlew = origw//2
        middleh = origh//2

        imgs = [
            im[middlew - crop_width // 2:middlew + crop_width // 2,
            middleh - crop_height // 2:middleh + crop_height // 2]
            for im in imgs]

        if label_type == 'mask_image':
            labels = [
                im[middlew - crop_width // 2:middlew + crop_width // 2,
                middleh - crop_height // 2:middleh + crop_height // 2]
                for im in labels]


        return imgs, labels

    def crop(self, img, middlew, middleh, crop_width, crop_height):
        return img.crop((np.round(middlew - crop_width / 2. + 0.1).astype(int),
                                np.round(middleh - crop_height / 2. + 0.1).astype(int),
                                np.round(middlew + crop_width / 2. + 0.1).astype(int),
                                np.round(middleh + crop_height / 2. + 0.1).astype(int)))

    def rotate_random(self, imgs, labels, value):
        label_type = self.meta_data['label_type']
        method = self.meta_data['random_rotate_method']
        origh, origw, __ = imgs[0].shape

        #todo: added this during run. make sure it works!
        for i in range(len(imgs)):
            if method == 'uniform':
                rot_degree = np.round(np.random.uniform(-value, value))
            elif method == 'normal':
                rot_degree = np.round(np.random.normal(0, value))
            # capping rotation to 2*std
            if np.abs(rot_degree) > 2*value:
               rot_degree = np.sign(rot_degree)*2 * value

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
        origh, origw, __ = imgs[0].shape

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