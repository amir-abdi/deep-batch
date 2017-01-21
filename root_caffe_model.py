import socket
import data_handler
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import directory_settings as s
import constants as c
import myUtils

from layers import *

# sys.path.append(s.caffe_root + 'python')
import caffe
from caffe import layers as L
# sys.path.append(s.caffe_root + "examples/pycaffe/layers")
# sys.path.append(s.caffe_root + "examples/pycaffe")
import tools
caffe.set_mode_gpu()
caffe.set_device(0)

import json
from abc import ABCMeta, abstractmethod


class RootCaffeModel:
    """The abstract class RootCaffeModel.
    Overriding the train_validate method is required.

    :return: caffe model
    """
    __metaclass__ = ABCMeta

    def __init__(self, external_meta_data=None):
        self.meta_data = None
        self.external_meta_data = external_meta_data
        self.set_directories()
        self.model()
        self.data_handler = data_handler.data_handler(self.meta_data['label_type'], self.meta_data['load_to_memory'])
        self.solver_prototxt()

        self._training_iteration = 0
        self._epoch = -1

    def set_directories(self):
        self.model_dir = s.framework_root + 'models/' + self.__class__.__name__ + '/'
        self.snapshot_dir = osp.join(self.model_dir, 'snapshots/')
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def model(self):
        workdir = self.model_dir
        with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
            n = self.initialize_net()
            netstr = self.finalize_net(n, 'train')
            f.write(netstr)
        with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
            n = self.initialize_net()
            netstr = self.finalize_net(n, 'valid')
            f.write(netstr)

    def get_meta_data(self):
        if self.meta_data is None:
            self.meta_data = self.create_meta_data()
            if self.external_meta_data is not None:
                self.meta_data.update(self.external_meta_data)
        return self.meta_data

    def solver_prototxt(self):
        solverprototxt = tools.CaffeSolver(trainnet_prototxt_path=osp.join(self.model_dir, "trainnet.prototxt"),
                                           testnet_prototxt_path=osp.join(self.model_dir, "valnet.prototxt"))

        solverprototxt.sp['display'] = self.meta_data['display']
        solverprototxt.sp['base_lr'] = self.meta_data['base_lr']
        solverprototxt.sp['gamma'] = self.meta_data['gamma']
        solverprototxt.sp['lr_policy'] = self.meta_data['lr_policy']
        solverprototxt.sp['max_iter'] = self.meta_data['max_iter']
        solverprototxt.sp['momentum'] = self.meta_data['momentum']
        solverprototxt.sp['weight_decay'] = self.meta_data['weight_decay']
        solverprototxt.sp['solver_mode'] = self.meta_data['solver_mode']
        solverprototxt.sp['stepsize'] = self.meta_data['stepsize']
        solverprototxt.sp['gamma'] = self.meta_data['gamma']
        solverprototxt.sp['snapshot_prefix'] = '\"' + self.snapshot_dir + '\"'
        solverprototxt.sp['test_iter'] = '1'

        solverprototxt.sp['snapshot'] = '1000000'  # fake value
        solverprototxt.sp['test_interval'] = '1000000'  # fake value

        solverprototxt.write(osp.join(self.model_dir, 'solver.prototxt'))


    def set_data(self, train_list_file=None, valid_list_file=None, train_folder=None, valid_folder=None,
                 data=None):
        self.data_handler.set_data(train_list_file=train_list_file, valid_list_file=valid_list_file,
                                   train_folder=train_folder, valid_folder=valid_folder,
                                   data=data,
                                   image_format=self.meta_data['image_format'],
                                   split_ratio=self.meta_data['split_ratio'],
                                   load_to_memory=self.meta_data['load_to_memory'],
                                   subtract_mean=self.meta_data['subtract_mean'])

    def set_network_data(self, x, y, solver, train_valid):
        if train_valid == 'train':
            solver.net.blobs['data'].data[...] = x
            solver.net.blobs['label'].data[...] = y

        elif train_valid == 'valid':
            solver.test_nets[0].blobs['data'].data[...] = x

    def net_step(self, solver, train_valid):
        if train_valid == 'train':
            solver.step(1)
            return solver.net.blobs['loss'].data, solver.net.blobs['output'].data
        elif train_valid == 'valid':
            return solver.test_nets[0].forward()['output']

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

    def init_training_state(self):
        training_data = {'total_iterations': 0, 'best_validation_acc': -np.inf,
                         'train_acc': -np.inf, 'train_loss': np.inf, 'validation_loss': np.inf, 'validation_acc': -np.inf}
        plt.close()
        return training_data

    def is_end_training(self):
        window_w = self.meta_data['averaging_window']
        max_not_improve = self.meta_data['terminate_if_not_improved_epoch']
        min_epoch = self.meta_data['min_epoch']
        max_epoch = self.meta_data['max_epoch']
        if max_epoch is not None and self.epoch >= max_epoch:
            return True
        if self.epoch < min_epoch:
            return False
        if np.mean(self.training_history[  -window_w:         -1][c.VAL_ACCURACY]) > \
                np.mean(self.training_history[-2*window_w:-window_w-1][c.VAL_ACCURACY]) - 0.01:
            self.end_criteria_counter += 1
            if self.end_criteria_counter >= max_not_improve:
                print 'end of training criteria has reached'
                return True
        # else:
        #     self.end_criteria_counter = 0
        return False

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
        print 'wrote state and history files: {:s}.state, {:s}.history'.format(self.write_filename, self.write_filename)

    def update_training_state_training(self):
        self.training_state['train_acc'] = self.current_epoch_history[c.TRAIN_ACCURACY]
        self.training_state['train_loss'] = self.current_epoch_history[c.TRAIN_LOSS]

    def update_training_state_validation(self):
        self.training_state['validation_acc'] = self.current_epoch_history[c.VAL_ACCURACY]
        self.training_state['validation_loss'] = self.current_epoch_history[c.VAL_LOSS]

        if self.training_state['best_validation_acc'] < self.training_state['validation_acc']:
            self.training_state['best_validation_acc'] = self.training_state['validation_acc']
            self.training_state['best_validation_epoch'] = self.training_state['epoch']

    def snapshot_handler(self, solver, training_status):
        meta = self.meta_data
        if 'best' in meta['snapshot_approach'] and training_status['best_validation_epoch'] == \
                                                             training_status['epoch']:
            self.write_snapshot(solver, '_best')
        if 'step' in meta['snapshot_approach'] and self.epoch % meta['snapshot_epochs'] == 0:
            self.write_snapshot(solver, '_step')
        if 'last' in meta['snapshot_approach']:
            self.write_snapshot(solver, '_last')

        if 'step' in meta['snapshot_approach'] and self.epoch % meta['caffe_solver_state_epochs'] == 0:
            self.write_caffe_solver_state(solver)

    def write_snapshot(self, solver, type_str):
        solver.net.save(str(self.write_filename + type_str + '.caffemodel'))
        print 'Wrote snapshot to: {:s}.caffemodel'.format(self.write_filename)

    @staticmethod
    def write_caffe_solver_state(self, solver):
        solver.snapshot()

    def load_snapshot(self, weights_file, solver):
        #todo: not properly tested after refactoring
        print 'loading weights from ', weights_file
        solver.net.copy_from(weights_file)
        return solver

    def load_state(self, state_file):
        print 'loading state from ', state_file
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
        print 'loading training history from ', history_file
        t = np.load(history_file)
        t = t.tolist()
        return t

    def history(self):
        return self.history

    def save_plot_history(self, current_epoch):
        self.training_history.append(current_epoch)
        np.save(self.write_filename + '_history', np.asarray(self.training_history))
        myUtils.plot_show(np.asarray(self.training_history))
        plt.savefig(self.write_filename + '.png')

    def initialize_net(self):
        n = caffe.NetSpec()
        # this is just to fill the space
        n.data, n.label = L.MemoryData(batch_size=1, height=1, width=1, channels=1, ntop=2)
        return n

    def finalize_net(self, n, train_valid):
        meta_data = self.get_meta_data()
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

    def initialize_solver(self, solver_type):
        print "initializing solver..."
        if solver_type == 'SGD':
            solver = caffe.SGDSolver(osp.join(self.model_dir, 'solver.prototxt'))
            solver.test_nets[0].share_with(solver.net)
            return solver
        else:
            print 'Other solver types are not implemented. However, they are supposed to have the same signature'

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.solver = self.initialize_solver(solver_type)
        self.training_history = []
        if snapshot_weight is not None:
            solver = self.load_snapshot(snapshot_weight, self.solver)
        if snapshot_state is not None:
            self.training_state, self.hyper_meta_data, self.meta_data, data_handler_meta_data = self.load_state(snapshot_state)
            self.data_handler.set_meta_data_json(data_handler_meta_data)
        if snapshot_history is not None:
            self.training_history = self.load_training_history(snapshot_history)

        self.end_criteria_counter = 0
        self.write_filename = osp.join(self.snapshot_dir, self.meta_data['snapshot_str'])

        batch_size = self.meta_data['batch_size']
        train_size, valid_size = self.data_handler.get_dataset_size()
        self.nb_batches_train = np.round(train_size / batch_size)
        self.nb_batches_valid = np.round(valid_size / batch_size)

    def print_train_iteration(self, traini, loss_batch):
        if self.training_state['total_iterations'] % self.meta_data['display_iter'] == 0:
            print "epoch: ", self.epoch, "  train iteration: ", traini+1, "/", self.nb_batches_train, \
                ' batch_loss: ', loss_batch

    def print_valid_iteration(self, validi, loss_batch):
        if validi % self.meta_data['display_iter'] == 0:
            print "valid iteration: ", validi+1, "/", self.nb_batches_valid, \
                ' batch_loss: ', loss_batch

    def print_current_epoch_history(self):
        print "Training loss: {0:.3f}".format(self.current_epoch_history[c.TRAIN_LOSS]), \
            " train accuracy: {0:.3f}".format(self.current_epoch_history[c.TRAIN_ACCURACY])
        print "Validation loss: {0:.3f}".format(self.current_epoch_history[c.VAL_LOSS]), \
            " Validation accuracy: {0:.3f}".format(self.current_epoch_history[c.VAL_ACCURACY])
        print "=" * 120

    def get_solver(self):
        return self.solver

    @abstractmethod
    def train_validate(self):
        pass

    @abstractmethod
    def create_meta_data(self):
        pass

    @abstractmethod
    def net(self, train_valid):
        return "Define your network here"

    @property
    def total_training_iteration(self):
        return self._training_iteration

    @total_training_iteration.setter
    def total_training_iteration(self, value):
        self._training_iteration = value
        self.training_state['total_iterations'] = value
        epoch = np.round(value / self.nb_batches_train)
        if epoch != self.epoch:
            self.epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.training_state['epoch'] = value


