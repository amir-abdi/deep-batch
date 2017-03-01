import math

import numpy as np

from root_models.caffe.RootCaffeModel import RootCaffeModel
from root_models.caffe.layers import *
from utilities import constants as c


class DemoNetModel(RootCaffeModel):
    def __init__(self, external_params=None):
        super(DemoNetModel, self).__init__(external_params)

    def net(self, n, train_valid='train'):
        meta_data = self.get_meta_data()

        n.conv1, n.relu1 = conv_relu(n.data, 24, 21, 1)
        n.pool1 = max_pool(n.relu1, 5, 3)
        n.conv2, n.relu2 = conv_relu(n.pool1, 48, 7, 1)
        n.pool2 = max_pool(n.relu2, 3, 2)

        n.fc1, n.relu_fc1 = fc_relu(n.pool2, 512)
        n.dropout1 = L.Dropout(n.relu_fc1, in_place=True, dropout_param=dict(dropout_ratio=0.5))
        n.fc2, n.relu_fc2 = fc_relu(n.dropout1, 128)
        n.dropout2 = L.Dropout(n.relu_fc2, in_place=True, dropout_param=dict(dropout_ratio=0.5))
        n.output = fc(n.dropout2, nout=1, bias_constant=0)
        if train_valid == 'train':
            n.loss = L.EuclideanLoss(n.output, n.label)

        prototxt_str = str(n.to_proto())

        if train_valid == 'train':
            prototxt_str = data_label(prototxt_str,
                                      meta_data['batch_size'],
                                      meta_data['channels'],
                                      meta_data['im_height'],
                                      meta_data['im_width'],
                                      meta_data['label_type'])
        elif train_valid == 'valid':
            prototxt_str = data(prototxt_str,
                                meta_data['batch_size'],
                                meta_data['channels'],
                                meta_data['im_height'],
                                meta_data['im_width'],
                                meta_data['label_type'])
        return prototxt_str

    def create_meta_data(self):
        # todo: save learning rate in snapshot state, and load it. calculate learning rate after each iteration
        m = {
            'model_variant': 'DemoNet',
            'batch_size': 3,
            'channels': 1,
            'im_width': 267,
            'im_height': 267,
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
            'resize_width': 400,
            'resize_height': 267,
            'random_translate_std_ratio': 20,
            'random_rotate_degree': 7,
            'train_batch_method': 'uniform',  # 'random', 'uniform'
            'split_ratio': 0.1,  # set to 0 if not splitting train and valid
            'load_to_memory': True,
            'subtract_mean': False,
            'image_format': '.jpg',

            # end of training parameters
            'max_epoch': 60,
            'min_epoch': 30,
            'terminate_if_not_improved_epoch': 5,
            'averaging_window': 15
        }
        return m

    def train_validate(self):
        meta_data = self.meta_data
        self.training_state = self.init_training_state()
        solver = self.get_solver()

        batch_size = meta_data['batch_size']
        nb_batches_train = self.nb_batches_train
        nb_batches_valid = self.nb_batches_valid
        self.total_training_iteration = self.training_state['total_iterations']

        while not self.is_end_training():
            print("=" * 80)
            self.current_epoch_history = np.zeros(4)

            #training epoch loop
            for traini in range(nb_batches_train):
                x, y = self.data_handler.get_batch(batch_size=batch_size,
                                                   train_valid='train',
                                                   interclass_selection_method=meta_data['train_batch_method'])
                x, y = self.data_handler.preprocess(data_batch=x, label_batch=y,
                                                    rotate_degree=meta_data['random_rotate_degree'],
                                                    translate_std_ratio=meta_data['random_translate_std_ratio'],
                                                    crop_width=meta_data['im_width'],
                                                    crop_height=meta_data['im_height'],
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
                                                        crop_width=meta_data['im_width'],
                                                        crop_height=meta_data['im_height'],
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

            print("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)  # needs to be before epoch update to keep track of 'best_validation'

            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]