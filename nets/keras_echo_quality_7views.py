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


class KerasEchoQuality7Views(RootKerasModel):
    def __init__(self, external_dict=None):
        super().__init__(external_dict)

    def create_meta_data(self):
        # todo: save learning rate in snapshot state, and load it. calculate learning rate after each iteration
        m = {
            'model_variant': '7view_keras',
            'batch_size': 40,  # all views together

            # SGD
            'base_lr': 0.01,
            'lr_decay': 0.9,
            'momentum': 0.95,
            'weight_decay': 0.05,

            # display
            'display_iter': 1,

            # snapshot parameters
            'snapshot_fld': self.snapshot_dir,
            'model_fld': self.model_dir,
            'snapshot_approach': ['best', 'last'],  # 'last', 'best', 'step'
            'snapshot_epochs': 1,  # only used if 'step' is included in snapshot approach
            'caffe_solver_state_epochs': 1000,  # only used if 'step' is included in snapshot approach; handled by caffe
            'snapshot_str': self.__class__.__name__,  # update snapshot_str in the external metadata or here to whatever

            # validation
            'test_interval': 1,
            'test_approach': 'epoch',  # 'epoch', 'iter', 'none'; by setting as 'epoch', 'test_interval' is ignored

            # preprocess
            'resize_width': 200,
            'resize_height': 200,
            'crop_width': 200,
            'crop_height': 200,
            'channels': 1,
            'random_rotate_method': 'uniform',  # 'uniform', 'normal'
            'random_translate_method': 'uniform',  # 'uniform', 'normal'
            'random_translate_ratio_value': 15,
            'random_rotate_value': 7,

            # data handler parameters
            'train_intraclass_selection': 'uniform',
        # 'random', 'uniform'   [only applies if train_batch_method is random]
            'train_batch_method': 'random',  # 'iterative', 'random'
            'split_ratio': 0.1,  # set to 0 if not splitting train and valid
            'load_to_memory': False,
            'subtract_mean': False,
            'file_format': 'mat',  # 'mat', 'image'
            'delimiter': ',',
            'main_label_index': 0,
            'label_type': 'single_value',  # 'single_value', 'mask_image'
            'scale_label': 1,  # 0: do not rescale, else: rescale all labels to the value

            'multi_cine_per_patient': True,
            'cine_selection_if_not_multi': 'random',
        # 'random', 'first'  [only applies if multi_cine_per_patient is false]

            # end of training parameters
            'max_epoch': 100,
            'min_epoch': 50,
            'terminate_if_not_improved_epoch': 10,
            'averaging_window': 15,

            # cine
            'num_frames': 20

        }
        return m

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
            self.current_epoch_history = np.zeros(4+num_views)

            #training epoch loop
            for traini in range(nb_batches_train):
                x = []
                y = []
                input_args = dict()
                pred_args = dict()
                for view in range(num_views):
                    x_temp, y_temp = self.data_handler.get_batch(batch_size=batch_size_per_view,
                                                                 train_valid='train',
                                                                 batch_selection_method=meta_data['train_batch_method'],
                                                                 interclass_selection_method=meta_data['train_intraclass_selection'],
                                                                 view=view)
                    x_temp, y_temp = self.data_handler.preprocess(data_batch=x_temp, label_batch=y_temp,
                                                        rotate_degree=meta_data['random_rotate_value'],
                                                        translate_std_ratio=meta_data['random_translate_ratio_value'],
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


            lr = K.get_value(self.net_model.optimizer.lr)
            print('learning rate: {}'.format(lr))
            # K.set_value(self.net_model.optimizer.lr, lr * self.meta_data['lr_decay'])

            #validation loop
            if self.is_validation_epoch():
                input_args = dict()
                for validi in range(nb_batches_valid):
                    x = []
                    y = []
                    for view in range(num_views):
                        x_temp, y_temp = self.data_handler.get_batch(batch_size=batch_size_per_view,
                                                                                 train_valid='valid',
                                                                                    # batch_selection_method='iterative',
                                                                     batch_selection_method=meta_data['train_batch_method'],
                                                                     interclass_selection_method=meta_data['train_intraclass_selection'],
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
                    for t in range(num_views):
                        if self.meta_data['scale_label'] == 0:
                            self.current_epoch_history[4 + t] += ((self.meta_data['range_views'][t]-output[num_views + 1 + t])
                                                                  / self.meta_data['range_views'][t]
                                                                  / nb_batches_valid)
                        else:
                            self.current_epoch_history[4 + t] += ((self.meta_data['scale_label']-output[num_views + 1 + t]) / nb_batches_valid)
                    self.print_valid_iteration(validi, output, nb_batches_valid)

                print("End of Validation\n", "-" * 80)
                self.update_training_state_validation()

            print("End of Training Epoch\n", "-" * 80)
            self.update_training_state_training()
            self.snapshot_handler(solver, self.training_state)  # needs to be before epoch update to keep track of 'best_validation'


            self.save_show_plot_history(self.current_epoch_history)
            self.save_state(self.training_state)

            self.print_current_epoch_history()
        return self.training_history[-1][c.VAL_ACCURACY]

    def evaluate(self, weight_file):
        net_model = load_model(weight_file)
        print('model loaded from file: {:s}'.format(weight_file))
        meta_data = self.meta_data
        num_views = self.number_of_views
        batch_size = self.number_of_views
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
                                                             batch_selection_method='iterative',
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
                input_args.update({'input' + str(view): np.asarray(x_temp, )})
                pred_args.update({'pred' + str(view): np.asarray(y_temp, )})


            output = net_model.test_on_batch(input_args, pred_args)
            start_time = time.time()
            pred = net_model.predict_on_batch(input_args)
            end_time = time.time()
            print(end_time-start_time)

            # print(1, pred)
            # print(2, y)
            # print(3, output[num_views + 1:])
            # pred = net_model.predict(input_args)

            loss_batch = sum(output[1:num_views]) / num_views
            acc_batch = self.calculate_accuracy_from_absErr(output[num_views + 1:])
            loss += (loss_batch / nb_batches_test)
            acc += (acc_batch / nb_batches_test)

            accuracy[testi, :] = output[num_views + 1:]
            if self.meta_data['scale_label'] != 0:
                predictions[testi, :] = np.multiply(np.reshape(np.asarray(pred), num_views), self.meta_data['range_views']/self.meta_data['scale_label'])
                groundT[testi, :] = np.multiply(np.reshape(np.asarray(y), num_views), self.meta_data['range_views']/self.meta_data['scale_label'])
            else:
                predictions[testi, :] = np.reshape(np.asarray(pred), num_views)
                groundT[testi, :] = np.reshape(np.asarray(y), num_views)

            self.print_valid_iteration(testi, output, nb_batches_test)

        predictions[-1, :] = self.data_handler.get_testset_size_per_view()
        print('acc: {}'.format(acc), '   loss: {}'.format(loss))
        print('accuracy accros views: {}'.format(accuracy.mean(0)))
        np.savetxt(self.write_filename + "_accuracy.csv", accuracy, delimiter=',')
        np.savetxt(self.write_filename +'_pred.csv', predictions, delimiter=',')
        np.savetxt(self.write_filename +'_GT.csv', groundT, delimiter=',')

    def net(self):
        #todo: dropout
        num_frames = self.meta_data['num_frames']
        input_list = []
        for i in range(self.number_of_views):
            input_list.append(
                Input(shape=( self.meta_data['num_frames'],
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
        for i in range(self.number_of_views):
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

            # v[i] = LSTM(output_dim=32, return_sequences=False, activation='tanh')(v[i])
            # v[i] = Dense(1, activation='hard_sigmoid', name='pred'+str(i))(v[i])

            v[i] = LSTM(output_dim=1, return_sequences=False, activation='hard_sigmoid', name='pred'+str(i))(v[i])

            pred_list.append(v[i])

        self.net_model = Model(input=input_list, output=pred_list)

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.model()
        net_model = self.net_model
        # todo: multiple GPUs

        epochs = self.meta_data['max_epoch']
        lrate = 0.0002
        # decay = lrate / epochs
        # opt = SGD(lr=self.meta_data['base_lr'],
        #               momentum=self.meta_data['momentum'],
        #               # decay=decay,  # self.meta_data['weight_decay'],
        #               nesterov=True)


        opt = adam()

        net_model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])

        if snapshot_weight is not None:
            print('load snapshot for keras')
            # todo: load snapshot keras
        if snapshot_state is not None:
            self.training_state, self.hyper_meta_data, self.meta_data, data_handler_meta_data = self.load_state(
                snapshot_state)
            self.data_handler.set_meta_data_json(data_handler_meta_data)

        if snapshot_history is not None:
            self.training_history = self.load_training_history(snapshot_history)
