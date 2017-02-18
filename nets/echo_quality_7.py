import math
# from __builtin__ import setattr

import numpy as np
from caffe import layers as L

import constants as c
from layers import *
from RootCaffeModel import RootCaffeModel

num_views = 7

class EchoNet7(RootCaffeModel):
    def __init__(self, external_params=None):
        super(EchoNet7, self).__init__(external_params)

    def net(self, n, train_valid='train'):
        #shared layers
        n.conv1, n.relu1 = conv_relu(n.data, 10, 25, 1, same_size=True)
        n.pool1 = max_pool(n.relu1, 3, 2)
        n.conv2, n.relu2 = conv_relu(n.pool1, 20, 11, 1, same_size=True)
        n.pool2 = max_pool(n.relu2, 3, 2)
        n.conv3, n.relu3 = conv_relu(n.pool2, 30, 11, 1, same_size=True)
        n.pool3 = max_pool(n.relu3, 3, 2)

        #view specific layers
        #todo: two ways to do this: share weights, or what I am doing below!
        for i in range(num_views):
            setattr(n, 'conv4_'+str(i), conv(n.pool3, 40, 5, 1, same_size=True))
            setattr(n, 'relu4_'+str(i), L.ReLU(getattr(n, 'conv4_'+str(i)), in_place=True))
            setattr(n, 'pool4_' + str(i), max_pool(getattr(n, 'relu4_'+str(i)), 3, 2))

            setattr(n, 'conv5_' + str(i), conv(getattr(n, 'pool4_'+str(i)), 50, 3, 1, same_size=True))
            setattr(n, 'relu5_' + str(i), L.ReLU(getattr(n, 'conv5_' + str(i)), in_place=True))
            setattr(n, 'pool5_' + str(i), max_pool(getattr(n, 'relu5_' + str(i)), 3, 2))

            setattr(n, 'fc1_' + str(i), fc(getattr(n, 'pool5_'+str(i)), 512))
            setattr(n, 'relufc1_' + str(i), L.ReLU(getattr(n, 'fc1_' + str(i)), in_place=True))
            setattr(n, 'output_' + str(i), fc(getattr(n, 'relufc1_' + str(i)), 1))

            # # conv4 , n.relu4 = conv_relu(n.pool3, 40, 5, 1, same_size=True)
            # n.pool4 = max_pool(n.relu4, 3, 2)
            # n.conv5, n.relu5 = conv_relu(n.pool4, 50, 3, 1, same_size=True)
            # n.pool5 = max_pool(n.relu5, 3, 2)
            # n.output = fc(n.pool5, 512)
            # # n.output = L.Dropout(n.relu_fc1, in_place=True, dropout_param=dict(dropout_ratio=0.5))
            if train_valid == 'train':
                setattr(n, 'loss_' + str(i), L.EuclideanLoss(getattr(n, 'output_' + str(i)), n.label))
        return n

    def create_meta_data(self):
        # todo: save learning rate in snapshot state, and load it. calculate learning rate after each iteration
        m = {
            'model_variant': 'DemoNet',
            'batch_size': 3,
            'channels': 1,
            'crop_width': 267,
            'crop_height': 267,
            'base_lr': '0.0002',
            'display': '100',
            'type': '\"AdaGrad\"',
            'lr_policy': '\"step\"',
            'stepsize': '1000',
            'gamma': '0.5',
            'max_iter': '1000000',
            'momentum': '0.95',

            'weight_decay': '0.05',
            'regularization_type': '\'"L2\"',

            'solver_mode': 'GPU',

            # snapshot parameters
            'snapshot_fld': self.snapshot_dir,
            'model_fld': self.model_dir,
            'snapshot_approach': ['best', 'last'],  # 'last', 'best', 'step'
            'snapshot_epochs': 100,  # only used if 'step' is included in snapshot approach
            'caffe_solver_state_epochs': 1000,  # only used if 'step' is included in snapshot approach; handled by caffe
            'snapshot_str': self.__class__.__name__,  # update snapshot_str in the external metadata or here to whatever

            'test_interval': 1,
            'test_approach': 'epoch',  # 'epoch', 'iter', 'none'; by setting as 'epoch', 'test_interval' is ignored
            'label_type': 'single_value',  # 'single_value', 'mask_image'
            'display_iter': 1,

            # data handler parameters
            'resize_width': 400,
            'resize_height': 267,
            'random_translate_std_ratio': 20,
            'random_rotate_degree': 7,
            'train_batch_method': 'uniform',  # 'random', 'uniform'
            'split_ratio': 0.1,  # set to 0 if not splitting train and valid
            'load_to_memory': True,
            'subtract_mean': False,
            'file_format': 'mat',
            'delimiter': ',',

            # end of training parameters
            'max_epoch': 60,
            'min_epoch': 30,
            'terminate_if_not_improved_epoch': 5,
            'averaging_window': 15

        }
        return m

    def train_validate(self):
        meta_data = self.meta_data
        solver = self.get_solver()

        batch_size = meta_data['batch_size']
        nb_batches_train = self.nb_batches_train
        nb_batches_valid = self.nb_batches_valid
        self.total_training_iteration = self.training_state['total_iterations']

        while not self.is_end_training():
            print('=' * 80)
            self.current_epoch_history = np.zeros(4)

            #training epoch loop
            for traini in range(nb_batches_train):
                x, y = self.data_handler.get_batch(batch_size=batch_size,
                                                   train_valid='train',
                                                   interclass_selection_method=meta_data['train_batch_method'])
                x, y = self.data_handler.preprocess(data_batch=x, label_batch=y,
                                                    rotate_degree=meta_data['random_rotate_degree'],
                                                    translate_std_ratio=meta_data['random_translate_std_ratio'],
                                                    crop_width=meta_data['crop_width'],
                                                    crop_height=meta_data['crop_height'],
                                                    resize_width=meta_data['resize_width'],
                                                    resize_height=meta_data['resize_height'],
                                                    normalize_to_1_scale=False)

                self.set_network_data(x, y, solver, 'train')
                loss_batch, out = self.net_step(solver, 'train')
                acc_batch = -np.average(np.abs(out.T - y)) # calculate the accuracy of your batch however you see fit
                # if nan returned, there is something wrong with caffe and the model;terminate
                if math.isnan(loss_batch):
                    return np.inf

                self.current_epoch_history[c.TRAIN_LOSS] += (loss_batch / nb_batches_train)
                self.current_epoch_history[c.TRAIN_ACCURACY] += (acc_batch / nb_batches_train)
                self.print_train_iteration(traini, loss_batch)
                self.total_training_iteration += 1

            #validation loop
            if self.is_validation_epoch():
                for validi in range(nb_batches_valid):
                    x, y = self.data_handler.get_data_batch_iterative(batch_size=batch_size, train_valid='valid')
                    x, y = self.data_handler.preprocess(data_batch=x, label_batch=y,
                                                        crop_width=meta_data['crop_width'],
                                                        crop_height=meta_data['crop_height'],
                                                        resize_width=meta_data['resize_width'],
                                                        resize_height=meta_data['resize_height'],
                                                        normalize_to_1_scale=False)
                    self.set_network_data(x, y, solver, 'valid')
                    out = self.net_step(solver, 'valid')
                    loss_batch = np.average(np.power(out.T - y, 2))
                    acc_batch = -np.average(np.abs(out.T - y))  # added negative for accuracy to be meaningful
                    # if nan returned, there is something wrong with caffe and the model;terminate
                    if math.isnan(loss_batch):
                        return np.inf

                    self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    self.print_valid_iteration(validi, loss_batch)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print ("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)  # needs to be before epoch update to keep track of 'best_validation'

            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]