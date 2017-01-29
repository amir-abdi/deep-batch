from datahandler import DataHandler
import os
import directory_settings as s
import os.path as osp
from abc import ABCMeta, abstractmethod
import json
import myUtils
import numpy as np
import matplotlib.pyplot as plt
import constants as c

class RootModel:
    __metaclass__ = ABCMeta

    def __init__(self, external_meta_data=None):
        self.external_meta_data = external_meta_data
        self.set_directories()
        self.meta_data = None
        self.get_meta_data()

        self.model()
        self.data_handler = DataHandler()
        self.solver()

        self.training_state = self.init_training_state()

    def set_directories(self):
        self.model_dir = s.framework_root + 'models/' + self.__class__.__name__ + '/'
        self.snapshot_dir = osp.join(self.model_dir, 'snapshots/')
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def set_data(self, train_list_file=None, valid_list_file=None, train_folder=None, valid_folder=None,
                 data=None, delimeter=' '):

        self.meta_data.update({  'train_list': train_list_file,
                                    'train_folder': train_folder,
                                    'valid_list': valid_list_file,
                                    'valid_folder': valid_folder
                                    })
        self.data_handler.set_data(data, self.meta_data)
        batch_size = self.meta_data['batch_size']
        train_size, valid_size = self.data_handler.get_dataset_size()
        self.nb_batches_train = int(np.round(train_size / batch_size))
        self.nb_batches_valid = int(np.round(valid_size / batch_size))

    def get_meta_data(self):
        if self.meta_data is None:
            self.meta_data = self.create_meta_data()
            if self.external_meta_data is not None:
                self.meta_data.update(self.external_meta_data)
        return self.meta_data

    def load_state(self, state_file):
        print('loading state from ', state_file)
        with open(state_file, 'r') as f:
            f.next().strip()
            training_state = json.loads(myUtils.read_json_block(f))
            f.next().strip()
            hyper_meta_data = json.loads(myUtils.read_json_block(f))
            f.next().strip()
            meta_data = json.loads(myUtils.read_json_block(f))
            f.next().strip()
            data_handler_meta_data = json.loads(myUtils.read_json_block(f))
        return training_state, hyper_meta_data, meta_data, data_handler_meta_data

    def load_training_history(self, history_file):
        print('loading training history from ', history_file)
        t = np.load(history_file)
        t = t.tolist()
        return t

    def init_training_state(self):
        self._training_iteration = 0
        self._epoch = -1
        self.training_history = []
        self.end_criteria_counter = 0  # todo: this has to move into some part of history or log
        self.write_filename = osp.join(self.snapshot_dir, self.meta_data['snapshot_str'])
        training_data = {'total_iterations': 0, 'best_validation_acc': -np.inf,
                         'train_acc': -np.inf, 'train_loss': np.inf, 'validation_loss': np.inf, 'validation_acc': -np.inf}
        plt.close()

        self.window_w = self.meta_data['averaging_window'] if 'averaging_window' in self.meta_data else None
        self.max_not_improve = self.meta_data['terminate_if_not_improved_epoch'] \
            if 'terminate_if_not_improved_epoch' in self.meta_data else None
        self.min_epoch = self.meta_data['min_epoch'] if 'min_epoch' in self.meta_data else 0
        self.max_epoch = self.meta_data['max_epoch'] if 'max_epoch' in self.meta_data else int('inf')

        return training_data

    def is_end_training(self):
        max_epoch = self.max_epoch
        min_epoch = self.min_epoch
        window_w = self.window_w
        max_not_improve = self.max_not_improve
        if max_epoch is not None and self.epoch >= max_epoch:
            return True
        if self.epoch < min_epoch:
            return False
        if np.mean(self.training_history[  -window_w:         -1][c.VAL_ACCURACY]) > \
                np.mean(self.training_history[-2*window_w:-window_w-1][c.VAL_ACCURACY]) - 0.01:
            self.end_criteria_counter += 1
            if self.end_criteria_counter >= max_not_improve:
                print('end of training criteria has reached')
                return True
        # else:
        #     self.end_criteria_counter = 0
        return False

    def print_train_iteration(self, traini, loss_batch):
        if self.training_state['total_iterations'] % self.meta_data['display_iter'] == 0:
            print("epoch: ", self.epoch, "  train iteration: ", traini+1, "/", self.nb_batches_train, \
                ' batch_loss: ', loss_batch)

    def print_valid_iteration(self, validi, loss_batch):
        if validi % self.meta_data['display_iter'] == 0:
            print("valid iteration: ", validi+1, "/", self.nb_batches_valid, \
                ' batch_loss: ', loss_batch)

    def print_current_epoch_history(self):
        print("Training loss: {0:.3f}".format(self.current_epoch_history[c.TRAIN_LOSS]), \
            " train accuracy: {0:.3f}".format(self.current_epoch_history[c.TRAIN_ACCURACY]))
        print("Validation loss: {0:.3f}".format(self.current_epoch_history[c.VAL_LOSS]), \
            " Validation accuracy: {0:.3f}".format(self.current_epoch_history[c.VAL_ACCURACY]))
        print("=" * 120)

    def is_validation_epoch(self):
        nb_batches = self.nb_batches_train
        total_iterations = self.total_training_iteration
        meta_data = self.meta_data

        if meta_data['test_approach'] == 'epoch' and total_iterations % nb_batches == 0:
            return True
        if meta_data['test_approach'] == 'iter' and total_iterations % meta_data['test_interval'] == 0:
            return True
        if meta_data['test_approach'] == 'none':
            return False
        return False

    def update_training_state_training(self):
        self.training_state['train_acc'] = self.current_epoch_history[c.TRAIN_ACCURACY]
        self.training_state['train_loss'] = self.current_epoch_history[c.TRAIN_LOSS]

    def update_training_state_validation(self):
        self.training_state['validation_acc'] = self.current_epoch_history[c.VAL_ACCURACY]
        self.training_state['validation_loss'] = self.current_epoch_history[c.VAL_LOSS]

        if self.training_state['best_validation_acc'] < self.training_state['validation_acc']:
            self.training_state['best_validation_acc'] = self.training_state['validation_acc']
            self.training_state['best_validation_epoch'] = self.training_state['epoch']

    def save_plot_history(self, current_epoch):
        self.training_history.append(current_epoch)
        np.save(self.write_filename + '_history', np.asarray(self.training_history))
        myUtils.plot_show(np.asarray(self.training_history))
        plt.savefig(self.write_filename + '.png')

    def save_state(self, training_state):
        with open(self.write_filename + '.state', 'w') as f:
            f.write('training_status\n')
            json.dump(training_state, f, indent=2)
            f.write('\n')
            f.write('hyper_meta_data\n')
            json.dump(self.external_meta_data, f, indent=2)
            f.write('\n')
            f.write('meta_data\n')
            json.dump(self.meta_data, f, indent=2)
            f.write('\n')
            f.write('data_handler.meta_data\n')
            json.dump(self.data_handler.get_meta_data(), f, indent=2)
            f.write('\n')
        print('wrote state and history files: {:s}.state, {:s}.history'.format(self.write_filename, self.write_filename))

    @property
    def total_training_iteration(self):
        return self._training_iteration

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.training_state['epoch'] = value

    @total_training_iteration.setter
    def total_training_iteration(self, value):
        self._training_iteration = value
        self.training_state['total_iterations'] = value
        epoch = np.round(value / self.nb_batches_train)
        if epoch != self.epoch:
            self.epoch = epoch

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def solver(self):
        pass

    @abstractmethod
    def create_meta_data(self):
        pass

    @abstractmethod
    def set_solver(self):
        pass

    @abstractmethod
    def snapshot_handler(self, solver, training_status):
        pass

    @abstractmethod
    def write_snapshot(self, solver, type_str):
        pass