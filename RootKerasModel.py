import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.layers import Convolution2D, activations
from keras.layers.convolutional import MaxPooling2D
from RootModel import RootModel
from keras.models import Model
import numpy as np
import constants as c

class RootKerasModel(RootModel):
    def __init__(self, external_params=None):
        super().__init__(external_params)
        self.model()

    def model(self):
        self.net()

    def net(self):
        input1 = Input(shape=(self.meta_data['channels'],
                              self.meta_data['crop_height'],
                              self.meta_data['crop_width']), name='input1')
        input2 = Input(shape=(self.meta_data['channels'],
                              self.meta_data['crop_height'],
                              self.meta_data['crop_width']), name='input2')


        conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
        max1 = MaxPooling2D((3, 3), strides=(2, 2))
        conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')
        max2 = MaxPooling2D((3, 3), strides=(2, 2))
        conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')
        max3 = MaxPooling2D((3, 3), strides=(2, 2))

        pred1 = Dense(1, name='pred1')(input1)
        # b1 = conv1(input1)
        # b1 = max1(b1)
        # b1 = conv2(b1)
        # b1 = max2(b1)
        # b1 = conv3(b1)
        # b1 = max3(b1)

        b2 = conv1(input2)
        b2 = max1(b2)
        b2 = conv2(b2)
        b2 = max2(b2)
        b2 = conv3(b2)
        b2 = max3(b2)

        # b1 = Convolution2D(10, 9, 9, border_mode='same', input_shape=(1, 256, 256))(b1)
        # b1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(b1)
        # b1 = Dense(512, activation='relu')(b1)
        # pred1 = Dense(1, name='pred1')(b1)

        b2 = Convolution2D(10, 9, 9, border_mode='same', input_shape=(1, 256, 256))(b2)
        b2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(b2)
        b2 = Dense(512, activation='relu')(b2)
        pred2 = Dense(1, name='pred2')(b2)

        # net_model = Sequential()
        # x = (Convolution2D(10, 9, 9, border_mode='same', input_shape=(1, 256, 256)))(input_shape)
        # x = Activation('relu')(x)
        # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        #
        # x = Dense(512)(x)
        # prediction = Dense(1)(x)

        # self.net_model = Model(input=input_shape, output=prediction)
        self.net_model = Model(input=[input1, input2], output=[pred1, pred2])
        # self.net_model = Model(input=input1, output=pred1)

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        net_model = self.net_model
        net_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

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
            'model_variant': 'DemoNet',
            'batch_size': 4,
            'channels': 1,
            'crop_width': 400,
            'crop_height': 400,
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

            # end of training parameters
            'max_epoch': 60,
            'min_epoch': 30,
            'terminate_if_not_improved_epoch': 5,
            'averaging_window': 15

        }
        return m

    def get_solver(self):
        return self.net_model

    def train_validate(self):
        meta_data = self.meta_data
        solver = self.get_solver()

        batch_size = meta_data['batch_size']
        nb_batches_train = self.nb_batches_train
        nb_batches_valid = self.nb_batches_valid
        self.total_training_iteration = self.training_state['total_iterations']

        num_views = 2
        batch_size_per_view = batch_size // num_views
        while not self.is_end_training():
            print('=' * 80)
            self.current_epoch_history = np.zeros(4)

            #training epoch loop
            for traini in range(nb_batches_train):
                x = []
                y = []
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
                                                        normalize_to_1_scale=False,
                                                        add_unit_channel=True)
                    x.append(x_temp)
                    y.append(y_temp)

                loss_batch = self.net_model.train_on_batch({'input1': np.asarray(x[0]), 'input2': np.asarray(x[1])},
                                                           {'pred1': np.asarray(y[0]), 'pred2': np.asarray(y[1])})
                # loss_batch = self.net_model.train_on_batch(np.asarray(x[0]),np.asarray(y[0]))
                acc_batch = loss_batch  # todo: check if you can implement this; maybe with another output

                self.current_epoch_history[c.TRAIN_LOSS] += (loss_batch / nb_batches_train)
                self.current_epoch_history[c.TRAIN_ACCURACY] += (acc_batch / nb_batches_train)
                self.print_train_iteration(traini, loss_batch)
                self.total_training_iteration += 1

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
                                                            crop_width=meta_data['im_width'],
                                                            crop_height=meta_data['im_height'],
                                                            resize_width=meta_data['resize_width'],
                                                            resize_height=meta_data['resize_height'],
                                                            normalize_to_1_scale=False)
                        x.append(x_temp)
                        y.append(y_temp)

                    loss_batch = self.net_model.test_on_batch(x, y)
                    acc_batch = loss_batch  # todo: check if you can implement this; maybe with another output

                    self.current_epoch_history[c.VAL_ACCURACY] += (acc_batch / nb_batches_valid)
                    self.current_epoch_history[c.VAL_LOSS] += (loss_batch / nb_batches_valid)
                    self.print_valid_iteration(validi, loss_batch)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print ("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)  # needs to be before epoch update to keep track of 'best_validation'

            self.save_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]

    def snapshot_handler(self, solver, training_status):
        meta = self.meta_data
        if 'best' in meta['snapshot_approach'] and training_status['best_validation_epoch'] == \
                training_status['epoch']:
            self.write_snapshot(solver, '_best')
        if 'step' in meta['snapshot_approach'] and self.epoch % meta['snapshot_epochs'] == 0:
            self.write_snapshot(solver, '_step')
        if 'last' in meta['snapshot_approach']:
            self.write_snapshot(solver, '_last')

    def write_snapshot(self, solver, type_str):
        pass




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