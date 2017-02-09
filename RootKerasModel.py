import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.layers import Convolution2D, activations
from keras.layers.convolutional import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, adagrad, adadelta
from keras.layers import Flatten
from RootModel import RootModel
from keras.models import Model
import numpy as np
import constants as c
from keras import backend as K
import matplotlib.pyplot as plt
from keras.engine.topology import Layer

class RootKerasModel(RootModel):
    def __init__(self, external_dict=None):
        super().__init__(external_dict)

    def model(self):
        self.net()

    def net(self):
        #todo: dropout
        input_list = []
        for i in range(self.number_of_views):
            input_list.append(
                Input(shape=(
                              self.meta_data['crop_height'],
                              self.meta_data['crop_width'],
                              self.meta_data['channels']),
                              name='input'+str(i))
            )

        conv1 = Convolution2D(10, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1))
        max1 = MaxPooling2D((3, 3), strides=(2, 2))
        conv2 = Convolution2D(20, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1))
        max2 = MaxPooling2D((3, 3), strides=(2, 2))
        conv3 = Convolution2D(30, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1))
        max3 = MaxPooling2D((3, 3), strides=(2, 2))

        v = []
        pred_list = []
        for i in range(self.number_of_views):
            v.append(conv1(input_list[i]))
            v[i] = max1(v[i])
            v[i] = conv2(v[i])
            v[i] = max2(v[i])
            v[i] = conv3(v[i])
            v[i] = max3(v[i])

            v[i] = Convolution2D(40, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(1))(v[i])
            v[i] = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(v[i])
            v[i] = Flatten()(v[i])
            v[i] = Dense(512, activation='relu')(v[i])
            pred_list.append(Dense(1, name='pred'+str(i))(v[i]))

        self.net_model = Model(input=input_list, output=pred_list)

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.model()
        net_model = self.net_model
        #todo: multiple GPUs

        epochs = 25
        lrate = 0.00001
        decay = lrate / epochs
        # opt = SGD(lr=self.meta_data['base_lr'],
        #               momentum=self.meta_data['momentum'],
        #               decay=self.meta_data['weight_decay'],
        #               nesterov=False)

        opt = adadelta()

        net_model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])

        # net_model.compile(optimizer='rmsprop',
        #               loss={'pred1': 'mse', 'pred2': 'mse'},
        #               loss_weights={'pred1': 1., 'pred2': 1.})


        if snapshot_weight is not None:
            print('load snapshot for keras')
            #todo: load snapshot keras
        if snapshot_state is not None:
            self.training_state, self.hyper_meta_data, self.meta_data, data_handler_meta_data = self.load_state(snapshot_state)
            self.data_handler.set_meta_data_json(data_handler_meta_data)

        if snapshot_history is not None:
            self.training_history = self.load_training_history(snapshot_history)

    def create_meta_data(self):
        # todo: save learning rate in snapshot state, and load it. calculate learning rate after each iteration
        m = {
            'model_variant': '7view_keras',
            'batch_size': 14,
            'channels': 1,
            'crop_width': 400,
            'crop_height': 400,
            'base_lr': 0.00001,
            'display': '100',
            'type': '\"AdaGrad\"',
            'lr_policy': '\"step\"',
            'stepsize': '1000',
            'gamma': '0.5',
            'max_iter': '1000000',
            'momentum': 0.95,
            'weight_decay': 0.05,
            'regularization_type': '\'"L2\"',

            'solver_mode': 'GPU',
            'display_iter': 1,

            # snapshot parameters
            'snapshot_fld': self.snapshot_dir,
            'model_fld': self.model_dir,
            'snapshot_approach': ['best', 'last'],  # 'last', 'best', 'step'
            'snapshot_epochs': 1,  # only used if 'step' is included in snapshot approach
            'caffe_solver_state_epochs': 1000,  # only used if 'step' is included in snapshot approach; handled by caffe
            'snapshot_str': self.__class__.__name__,  # update snapshot_str in the external metadata or here to whatever

            #validation
            'test_interval': 1,
            'test_approach': 'epoch',  # 'epoch', 'iter', 'none'; by setting as 'epoch', 'test_interval' is ignored

            # data handler parameters
            'resize_width': 400,
            'resize_height': 400,
            'random_translate_std_ratio': 20,
            'random_rotate_degree': 7,
            'train_batch_method': 'random',  # 'random', 'uniform'
            'split_ratio': 0.1,  # set to 0 if not splitting train and valid
            'load_to_memory': False,
            'subtract_mean': False,
            'file_format': 'mat',  # 'mat', 'image'
            'delimiter': ',',
            'main_label_index': 0,
            'label_type': 'single_value',  # 'single_value', 'mask_image'

            # end of training parameters
            'max_epoch': 60,
            'min_epoch': 30,
            'terminate_if_not_improved_epoch': 5,
            'averaging_window': 15

        }
        return m

    def get_solver(self):
        return self.net_model

    def calculate_accuracy_from_absErr(self, abs_errs):
        range_views = self.meta_data['range_views']
        abs_errs = np.asarray(abs_errs)
        metric = np.mean((range_views - abs_errs) / range_views)
        return metric

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

        num_views = self.number_of_views
        batch_size_per_view = batch_size // num_views
        while not self.is_end_training():
            print('=' * 80)
            print('Start Training')
            # self.current_epoch_history = np.zeros((num_views, 4))
            self.current_epoch_history = np.zeros(4)

            #training epoch loop
            for traini in range(nb_batches_train):
                x = []
                y = []
                input_args = dict()
                pred_args = dict()
                for view in range(num_views):
                    x_temp, y_temp = self.data_handler.get_data_batch_random(batch_size=batch_size_per_view,
                                                                   train_valid='train',
                                                                   method=meta_data['train_batch_method'],
                                                                    view=view)
                    x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp,
                                                        rotate_degree=meta_data['random_rotate_degree'],
                                                        translate_std_ratio=meta_data['random_translate_std_ratio'],
                                                        crop_width=meta_data['crop_width'],
                                                        crop_height=meta_data['crop_height'],
                                                        resize_width=meta_data['resize_width'],
                                                        resize_height=meta_data['resize_height'],
                                                        normalize_to_1_scale=True,
                                                        add_unit_channel=True
                                                        )
                    x.append(x_temp)
                    y.append(y_temp)
                    input_args.update({'input' + str(view): np.asarray(x[view], )})
                    pred_args.update({'pred' + str(view): np.asarray(y[view], )})

                output = self.net_model.train_on_batch(input_args, pred_args)
                # loss_batch = self.net_model.train_on_batch({'input1': np.asarray(x[0]), 'input2': np.asarray(x[1])},
                #                                            {'pred1': np.asarray(y[0]), 'pred2': np.asarray(y[1])})
                # loss_batch = self.net_model.train_on_batch(np.asarray(x[0]),np.asarray(y[0]))
                # acc_batch = loss_batch  # todo: check if you can implement this; maybe with another output
                # acc_batch = [-x for x in acc_batch]
                # acc_batch = -sum(output[num_views+1:])/num_views
                loss_batch = sum(output[1:num_views])/num_views
                acc_batch = self.calculate_accuracy_from_absErr(output[num_views+1:])

                # for view in range(num_views):
                self.current_epoch_history[c.TRAIN_LOSS] += (loss_batch / nb_batches_train)
                self.current_epoch_history[c.TRAIN_ACCURACY] += (acc_batch / nb_batches_train)

                self.print_train_iteration(traini, output)
                self.total_training_iteration += 1
                print("epoch: ", self.epoch, "  train iteration: ", traini + 1, "/", self.nb_batches_train, \
                    ' average batch_loss: ', loss_batch, '   average batch acc: ', acc_batch)

            #validation loop
            if self.is_validation_epoch():
                for validi in range(nb_batches_valid):
                    x = []
                    y = []
                    for view in range(num_views):
                        x_temp, y_temp = self.data_handler.get_data_batch_random(batch_size=batch_size_per_view,
                                                                                 train_valid='train',
                                                                                 method=meta_data['train_batch_method'],
                                                                                 view=view)
                        x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp,
                                                            crop_width=meta_data['crop_width'],
                                                            crop_height=meta_data['crop_height'],
                                                            resize_width=meta_data['resize_width'],
                                                            resize_height=meta_data['resize_height'],
                                                            normalize_to_1_scale=True,
                                                            add_unit_channel=True
                                                            )
                        x.append(x_temp)
                        y.append(y_temp)
                        input_args.update({'input' + str(view): np.asarray(x[view], )})
                        pred_args.update({'pred' + str(view): np.asarray(y[view], )})

                    output = self.net_model.test_on_batch(input_args, pred_args)

                    # acc_batch = loss_batch  # todo: check if you can implement this; maybe with another output
                    # acc_batch = [-x for x in acc_batch]
                    # acc_batch = -sum(output[num_views + 1:]) / num_views
                    loss_batch = sum(output[1:num_views]) / num_views
                    acc_batch = self.calculate_accuracy_from_absErr(output[num_views + 1:])

                    # self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    # self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    # for view in range(num_views):
                    self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    self.print_valid_iteration(validi, output)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)  # needs to be before epoch update to keep track of 'best_validation'

            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]

    def write_snapshot(self, solver, type_str):
        file = str(self.write_filename + type_str + '.kerasmodel')
        solver.save(file)
        print('Wrote snapshot to: {:s}'.format(file))




#   model = Model(input=[main_input, auxiliary_input], output=[main_output, auxiliary_output])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',
#               loss_weights=[1., 0.2])
# model.compile(optimizer='rmsprop',
#               loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
#               loss_weights={'main_output': 1., 'aux_output': 0.2})
#
# # and trained it via:
# model.fit({'main_input': headline_data, 'aux_input': additional_data},
#           {'main_output': labels, 'aux_output': labels},
#           nb_epoch=50, batch_size=32)


 # net_model = Sequential()
        # x = (Convolution2D(10, 9, 9, border_mode='same', input_shape=(1, 256, 256)))(input_shape)
        # x = Activation('relu')(x)
        # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        #
        # x = Dense(512)(x)
        # prediction = Dense(1)(x)