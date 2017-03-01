import math
import numpy as np
from root_models.caffe.RootCaffeModel import RootCaffeModel
from root_models.caffe.mylayers import *
from utilities import constants as c
import matplotlib.pyplot as plt

class CaffeDemoClassificationNet(RootCaffeModel):
    def __init__(self, external_params=None):
        super(CaffeDemoClassificationNet, self).__init__(external_params)

    def init_meta_data(self):
        m = {
            # caffe-specific parameters
            'caffe_display': 100,
            'caffe_max_iter': 1000000,
            'caffe_weight_decay': 0.005,
            'caffe_lr_policy': 'fixed',
            'caffe_stepsize': 1000,  # [if solver is SGD]
            'caffe_gamma': 0.5,  # [if solver is SGD]

            # snapshot parameters
            'snapshot_fld': self.snapshot_dir,
            'model_fld': self.model_dir,
            'caffe_solver_state_epochs': 1000,  # only used if 'step' is included in snapshot approach; handled by caffe
            'snapshot_str': self.__class__.__name__,  # update snapshot_str in the external metadata or here to whatever

            # optimizer
            'solver': 'adam',
            'base_lr': 0.001,
            'solver_mode': 'GPU',
            'momentum': 0.9,
            'momentum2': 0.999,
            'regularization_type': 'L2',

            # display
            'display_iter': 1,  # display results after X iterations

            # snapshot parameters
            'snapshot_approach': ['best', 'last'],  # values={'last', 'best', 'step'}, use multiple if needed
            'snapshot_epochs': 1,  # [if step is included in snapshot approach] snapshot after X epochs

            # validation
            'test_approach': 'epoch',  # 'epoch', 'iter', 'none'; by setting as 'epoch', 'test_interval' is ignored
            'test_interval': 1,  # [if test_approach is iter] number of test iterations before running validation

            # end of training parameters
            'max_epoch': 10,  # maximum number of epochs to train
            'min_epoch': 5,  # minimum number of epochs to train
            'terminate_if_not_improved_epoch': 10,  # terminate if validation accuracy had this many decreasing patterns
            'averaging_window': 15,  # averaging window to calculate validation accuracy

            # cine
            'sequence_length': 1,  # sequence length

            # DataHandler: preprocessing
            'multi_stream': False,
            'batch_size': 40,
            'channels': 3,  # number of channels for sample image
            'resize_width': 200,  # resize input image width, prior to cropping
            'resize_height': 200,  # resize input image height, prior to cropping
            'crop_width': 200,  # crop the middle square with this width
            'crop_height': 200,  # crop the middle square with this height
            'random_rotate_method': 'uniform',  # values={'uniform', 'normal'}
            'random_rotate_value': 7,  # MeanValue of normal method; LimitValue of uniform method
            'random_translate_method': 'uniform',  # values={'uniform', 'normal'}
            'random_translate_ratio_value': 15,  # MeanValue of normal method; LimitValue of uniform method
            'scale_label': 0,  # values={0: do not rescale, else: rescale all labels by the given value}
            'stream_specific_scale_label': None,  # values={None, list} [if streams have different label ranges]
            'scale_data': 255.,  # values={1: do not rescale, else: rescale all labels by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'reshape_batch': 'caffe',

            # DataHandler: reading and preparing training-validation or test data
            'split_ratio': 0.1,  # splitting ratio for train-validation (set to 0 if not splitting train and valid)
            'file_format': 'mat',  # values={'mat', 'image'}
            'delimiter': ',',  # list_file delimiter
            'load_to_memory': False,  # load all training-validation or testing data into memory before training

            # DataHandler: label parameters
            'label_type': 'single_value',  # values={'single_value', 'mask_image'}
            'main_label_index': 0,  # [if list file has multiple label values, which label index to use for training]

            # DataHandler: batch selection strategies
            'interclass_batch_selection': 'uniform',  # [if train_interclass_batch_selection is random]; values={'random', 'uniform'}
            'data_traversing': 'random',  # values={'iterative', 'random'}
            'multi_cine_per_patient': True,  # extract multiple training sequences from a single input sample
            'cine_selection_if_not_multi': 'random'  # [if multi_cine_per_patient is false] values={'random', 'first'}
        }
        return m

    def net(self, n, train_valid='train'):
        meta_data = self.get_meta_data()

        n.conv1, n.relu1 = conv_relu(n.data, 24, 21, 1)
        n.pool1 = max_pool(n.relu1, 5, 3)
        n.conv2, n.relu2 = conv_relu(n.pool1, 48, 7, 1)
        n.pool2 = max_pool(n.relu2, 3, 2)

        n.fc1, n.relu_fc1 = fc_relu(n.pool2, 1024)
        n.dropout1 = layers.Dropout(n.relu_fc1, in_place=True, dropout_param=dict(dropout_ratio=0.5))
        n.fc2, n.relu_fc2 = fc_relu(n.dropout1, 256)
        n.output = fc(n.fc2, nout=10, bias_constant=0)  # 10 = number of classes
        if train_valid == 'train':
            n.loss = layers.SoftmaxWithLoss(n.output, n.label)
        # to use caffe implementation of accuracy layer, uncomment next two lines; else, implement your own accuracy
        # elif train_valid == 'valid':
        #     n.acc = layers.Accuracy(n.output, n.label)

        return n

    def train_validate(self):
        print('=' * 80)
        print('Initialize network...')
        meta_data = self.meta_data
        solver = self.get_solver()

        print('Initialize learning parameters...')
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
                                                   data_traversing=meta_data['data_traversing'])
                x, y = self.data_handler.preprocess(data_batch=x, label_batch=y)

                self.set_network_data(x, y, solver, 'train')
                loss_batch, out = self.net_step(solver, 'train')
                acc_batch = self.my_accuracy(out, y)

                # if nan returned by caffe, there is something wrong with caffe and the model;terminate
                if math.isnan(loss_batch):
                    return np.inf

                self.current_epoch_history[c.TRAIN_LOSS] += (loss_batch / nb_batches_train)
                self.current_epoch_history[c.TRAIN_ACCURACY] += (acc_batch / nb_batches_train)
                self.print_train_iteration(traini, loss_batch)
                self.total_training_iteration += 1

            #validation loop
            if self.is_validation_epoch():
                for validi in range(nb_batches_valid):
                    x, y = self.data_handler.get_batch(batch_size=batch_size,
                                                       train_valid='valid',
                                                       data_traversing='iterative')
                    x, y = self.data_handler.preprocess(data_batch=x, label_batch=y)
                    self.set_network_data(x, y, solver, 'valid')
                    out = self.net_step(solver, 'valid')
                    loss_batch = np.average(np.power(out.T - y, 2))
                    acc_batch = self.my_accuracy(out, y)

                    # if nan returned by caffe, there is something wrong with caffe and the model;terminate
                    if math.isnan(loss_batch):
                        return np.inf

                    self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    self.print_valid_iteration(validi, loss_batch, nb_batches_valid)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)

            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]

    def my_accuracy(self, pred, y):
        # calculate the accuracy of your batch however you see fit
        bs = pred.shape[0]
        return sum([1 if np.argmax(pred[i]) == y[i] else 0 for i in range(pred.shape[0])]) / bs
