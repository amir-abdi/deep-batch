import numpy as np
from keras import backend as K
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import load_model
from keras.optimizers import adam
from keras.regularizers import l2

from root_models.keras.RootKerasModel import RootKerasModel
from utilities import constants as c


class KerasDemoMultiStreamRegressionNet(RootKerasModel):
    def __init__(self, external_dict=None):
        super().__init__(external_dict)

    def init_meta_data(self):
        # todo: save learning rate in snapshot state, and load it. calculate learning rate after each iteration
        m = {
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
            'max_epoch': 100,  # maximum number of epochs to train
            'min_epoch': 50,  # minimum number of epochs to train
            'terminate_if_not_improved_epoch': 10,  # terminate if validation accuracy had this many decreasing patterns
            'averaging_window': 15,  # averaging window to calculate validation accuracy

            # cine
            'sequence_length': 3,  # values={0: no sequential data, else: fixed length of each sequence}

            # DataHandler: preprocessing
            'multi_stream': True,
            'batch_size': 40,
            'channels': 1,  # number of channels for sample image
            'resize_width': 200,  # resize input image width, prior to cropping
            'resize_height': 200,  # resize input image height, prior to cropping
            'crop_width': 200,  # crop the middle square with this width
            'crop_height': 200,  # crop the middle square with this height
            'random_rotate_method': 'uniform',  # values={'uniform', 'normal'}
            'random_rotate_value': 7,  # MeanValue of normal method; LimitValue of uniform method
            'random_translate_method': 'uniform',  # values={'uniform', 'normal'}
            'random_translate_ratio_value': 15,  # MeanValue of normal method; LimitValue of uniform method
            'scale_label': 1,  # values={0: do not rescale, else: rescale all labels by the given value}
            'stream_specific_scale_label': None,  # values={None, list} [if streams have different label ranges]
            'scale_data': 255.,  # values={1: do not rescale, else: rescale all labels by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'reshape_batch': 'tf',

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
            'cine_selection_if_not_multi': 'random',  # [if multi_cine_per_patient is false] values={'random', 'first'}
            'multi_cine_per_patient': True  # extract multiple training sequences from a single input sample
        }
        return m

    def net(self):
        #todo: dropout
        num_frames = self.meta_data['sequence_length']
        input_list = []
        for i in range(self.num_streams):
            input_list.append(
                Input(shape=( self.meta_data['sequence_length'],
                              self.meta_data['crop_height'],
                              self.meta_data['crop_width'],
                              self.meta_data['channels']),
                              name='input'+str(i))
            )
        conv1 = TimeDistributed(Convolution2D(8, 3, 3, activation='relu', border_mode='valid', init='normal', W_regularizer=l2(1)))
        max1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))
        conv2 = TimeDistributed(Convolution2D(16, 3, 3, activation='relu', border_mode='valid', init='normal', W_regularizer=l2(1)))
        max2 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))
        conv3 = TimeDistributed(Convolution2D(16, 3, 3, activation='relu', border_mode='valid', init='normal', W_regularizer=l2(1)))
        max3 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))
        v = []
        pred_list = []
        for i in range(self.num_streams):
            v.append(input_list[i])
            v[i] = conv1(v[i])
            v[i] = max1(v[i])
            v[i] = conv2(v[i])
            v[i] = max2(v[i])
            v[i] = conv3(v[i])
            v[i] = max3(v[i])
            v[i] = TimeDistributed(Dropout(0.5))(v[i])
            v[i] = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init='normal', W_regularizer=l2(1)))(v[i])
            v[i] = TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init='normal', W_regularizer=l2(1)))(v[i])
            v[i] = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(v[i])
            v[i] = TimeDistributed(Flatten())(v[i])
            v[i] = LSTM(output_dim=1, return_sequences=False, activation='hard_sigmoid', name='pred'+str(i))(v[i])
            pred_list.append(v[i])
        self.net_model = Model(input=input_list, output=pred_list)

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.model()
        net_model = self.net_model
        opt = adam()
        net_model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])

        if snapshot_weight is not None:
            print('load snapshot for keras')
            # todo: load snapshot keras
            raise NotImplementedError
        if snapshot_state is not None:
            self.training_state, self.hyper_meta_data, self.meta_data, data_handler_meta_data = self.load_state(
                snapshot_state)
            self.data_handler.set_meta_data_json(data_handler_meta_data)

        if snapshot_history is not None:
            self.training_history = self.load_training_history(snapshot_history)

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

        num_stream = self.num_streams
        batch_size_per_stream = batch_size // num_stream
        extra_history_values = num_stream if num_stream > 1 else 0
        label_ranges = self.meta_data['stream_specific_scale_label']
        while not self.is_end_training():
            print('=' * 80)
            print('Start Training')
            self.current_epoch_history = np.zeros(4 + extra_history_values)

            # training epoch loop
            for traini in range(nb_batches_train):
                x = []
                y = []
                input_args = dict()
                pred_args = dict()
                for view in range(num_stream):
                    x_temp, y_temp = self.data_handler.get_batch(batch_size=batch_size_per_stream,
                                                                 train_valid='train',
                                                                 data_traversing=meta_data['data_traversing'],
                                                                 stream_index=view)
                    x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp)
                    x.append(x_temp)
                    y.append(y_temp)
                    input_args.update({'input' + str(view): np.asarray(x[view], )})
                    pred_args.update({'pred' + str(view): np.asarray(y[view], )})

                output = self.net_model.train_on_batch(input_args, pred_args)

                loss_batch = sum(output[1:num_stream])/num_stream
                acc_batch = self.my_accuracy(output[num_stream + 1:])

                self.current_epoch_history[c.TRAIN_LOSS] += (loss_batch / nb_batches_train)
                self.current_epoch_history[c.TRAIN_ACCURACY] += (acc_batch / nb_batches_train)

                self.print_train_iteration(traini, output)
                self.total_training_iteration += 1
                print("epoch: ", self.epoch, "  train iteration: ", traini + 1, "/", self.nb_batches_train, \
                    ' average batch_loss: ', loss_batch, '   average batch acc: ', acc_batch)

            lr = K.get_value(self.net_model.optimizer.lr)
            print('learning rate: {}'.format(lr))
            # to manipulate the learning rate manually
            # K.set_value(self.net_model.optimizer.lr, lr * self.meta_data['lr_decay'])

            # validation loop
            if self.is_validation_epoch():
                input_args = dict()
                for validi in range(nb_batches_valid):
                    x = []
                    y = []
                    for view in range(num_stream):
                        x_temp, y_temp = self.data_handler.get_batch(batch_size=batch_size_per_stream,
                                                                     train_valid='valid',
                                                                     data_traversing='iterative',
                                                                     stream_index=view)
                        x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp)
                        x.append(x_temp)
                        y.append(y_temp)
                        input_args.update({'input' + str(view): np.asarray(x[view], )})
                        pred_args.update({'pred' + str(view): np.asarray(y[view], )})

                    output = self.net_model.test_on_batch(input_args, pred_args)

                    loss_batch = sum(output[1:num_stream]) / num_stream
                    acc_batch = self.my_accuracy(output[num_stream + 1:])

                    self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    for t in range(num_stream):
                        if self.meta_data['scale_label'] == 0:
                            self.current_epoch_history[4 + t] += ((label_ranges[t]-output[num_stream + 1 + t])
                                                                  / label_ranges[t]
                                                                  / nb_batches_valid)
                        else:
                            self.current_epoch_history[4 + t] += ((self.meta_data['scale_label']-output[num_stream + 1 + t]) / nb_batches_valid)
                    self.print_valid_iteration(validi, output, nb_batches_valid)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)

            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]

    def evaluate(self, weight_file):
        net_model = load_model(weight_file)
        print('model loaded from file: {:s}'.format(weight_file))
        meta_data = self.meta_data
        num_views = self.num_streams
        batch_size = self.num_streams
        nb_batches_test = self.nb_batches_test
        batch_size_per_view = batch_size // num_views

        loss = 0
        acc = 0
        accuracy = np.zeros((nb_batches_test+1, num_views))
        predictions = np.zeros((nb_batches_test+1, num_views))
        groundT = np.zeros((nb_batches_test+1, num_views))

        predictions[-1, :] = self.data_handler.get_testset_size_per_view()
        accuracy[-1, :] = self.data_handler.get_testset_size_per_view()
        groundT[-1, :] = self.data_handler.get_testset_size_per_view()

        for testi in range(nb_batches_test):
            x = []
            y = []

            input_args = dict()
            pred_args = dict()
            for view in range(num_views):
                x_temp, y_temp = self.data_handler.get_batch(batch_size=batch_size_per_view,
                                                             train_valid='test',
                                                             data_traversing='iterative',
                                                             stream_index=view)
                x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp,
                                                              crop_width=meta_data['crop_width'],
                                                              crop_height=meta_data['crop_height'],
                                                              resize_width=meta_data['resize_width'],
                                                              resize_height=meta_data['resize_height'],
                                                              scale_data=True,
                                                              add_unit_channel=True
                                                              )
                x.append(x_temp)
                y.append(y_temp)
                input_args.update({'input' + str(view): np.asarray(x_temp, )})
                pred_args.update({'pred' + str(view): np.asarray(y_temp, )})


            output = net_model.test_on_batch(input_args, pred_args)
            pred = net_model.predict_on_batch(input_args)

            loss_batch = sum(output[1:num_views]) / num_views
            acc_batch = self.my_accuracy(output[num_views + 1:])
            loss += (loss_batch / nb_batches_test)
            acc += (acc_batch / nb_batches_test)

            accuracy[testi, :] = output[num_views + 1:]
            self.print_valid_iteration(testi, output, nb_batches_test)

        predictions[-1, :] = self.data_handler.get_testset_size_per_view()
        print('acc: {}'.format(acc), '   loss: {}'.format(loss))
        print('accuracy accros views: {}'.format(accuracy.mean(0)))
        np.savetxt(self.write_filename + "_accuracy.csv", accuracy, delimiter=',')
        np.savetxt(self.write_filename +'_pred.csv', predictions, delimiter=',')
        np.savetxt(self.write_filename +'_GT.csv', groundT, delimiter=',')

    def my_accuracy(self, abs_errs):
        return 1/np.mean(abs_errs)
