import socket
import datahandler
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
import datahandler

import json
from abc import ABCMeta, abstractmethod
from RootModel import RootModel


class RootCaffeModel(RootModel):
    """The abstract class RootCaffeModel.
    Overriding the train_validate method is required.

    :return: caffe model
    """
    __metaclass__ = ABCMeta

    def __init__(self, external_params=None):
        super().__init__(external_params)

    def model(self):
        workdir = self.model_dir
        with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
            n = self.initialize_net()
            n = self.net(n, 'train')
            netstr = self.finalize_net(n, 'train')
            f.write(netstr)
        with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
            n = self.initialize_net()
            n = self.net(n, 'valid')
            netstr = self.finalize_net(n, 'valid')
            f.write(netstr)

    def initialize_net(self):
        n = caffe.NetSpec()
        # this is just to fill the space
        n.data, n.label = L.MemoryData(batch_size=1, height=1, width=1, channels=1, ntop=2)
        return n

    def finalize_net(self, n, train_valid):
        prototxt_str = str(n.to_proto())

        if train_valid == 'train':
            prototxt_str = data_label(prototxt_str,
                                      meta_data['batch_size'],
                                      meta_data['channels'],
                                      meta_data['crop_height'],
                                      meta_data['crop_width'],
                                      meta_data['label_type'])
        elif train_valid == 'valid':
            prototxt_str = data(prototxt_str,
                                meta_data['batch_size'],
                                meta_data['channels'],
                                meta_data['crop_height'],
                                meta_data['crop_width'],
                                meta_data['label_type'])
        return prototxt_str


    def solver(self):
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
        print('Wrote snapshot to: {:s}.caffemodel'.format(self.write_filename))

    @staticmethod
    def write_caffe_solver_state(self, solver):
        solver.snapshot()

    def load_snapshot(self, weights_file, solver):
        #todo: not properly tested after refactoring
        print('loading weights from ', weights_file)
        solver.net.copy_from(weights_file)
        return solver

    def history(self):
        return self.history

    def initialize_solver(self, solver_type):
        print("initializing solver...")
        if solver_type == 'SGD':
            solver = caffe.SGDSolver(osp.join(self.model_dir, 'solver.prototxt'))
            solver.test_nets[0].share_with(solver.net)
            return solver
        else:
            print('Other solver types are not implemented. However, they are supposed to have the same signature')

    def set_solver(self, solver_type='SGD', snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.solver = self.initialize_solver(solver_type)
        if snapshot_weight is not None:
            solver = self.load_snapshot(snapshot_weight, self.solver)
        if snapshot_state is not None:
            self.training_state, self.hyper_meta_data, self.meta_data, data_handler_meta_data = self.load_state(snapshot_state)
            self.data_handler.set_meta_data_json(data_handler_meta_data)
        if snapshot_history is not None:
            self.training_history = self.load_training_history(snapshot_history)


    def get_solver(self):
        return self.solver

    @abstractmethod
    def train_validate(self):
        pass

    @abstractmethod
    def net(self, n, train_valid):
        return "Define your network here"


