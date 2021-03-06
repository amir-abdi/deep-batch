from utilities import directory_settings
import os.path as osp
import sys
sys.path.append(directory_settings.caffe_root + 'examples/pycaffe/layers')  # the data layers folder
sys.path.append(directory_settings.caffe_root + 'examples/pycaffe')  # the tools folder
import caffe
import tools
from root_models.caffe.mylayers import *
from abc import ABCMeta, abstractmethod
from root_models.RootModel import RootModel


class RootCaffeModel(RootModel):
    """The abstract class RootCaffeModel.
    Overriding the train_validate method is required.

    :return: caffe model
    """
    __metaclass__ = ABCMeta

    def __init__(self, external_params=None):
        super().__init__(external_params)
        if self.meta_data['solver_mode'] == 'GPU':
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

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
        n.data, n.label = layers.MemoryData(batch_size=1, height=1, width=1, channels=1, ntop=2)
        return n

    def finalize_net(self, n, train_valid):
        prototxt_str = str(n.to_proto())

        if train_valid == 'train':
            prototxt_str = data_label(prototxt_str,
                                      self.meta_data['batch_size'],
                                      self.meta_data['channels'],
                                      self.meta_data['crop_height'],
                                      self.meta_data['crop_width'],
                                      self.meta_data['label_type'])
        elif train_valid == 'valid':
            prototxt_str = data(prototxt_str,
                                self.meta_data['batch_size'],
                                self.meta_data['channels'],
                                self.meta_data['crop_height'],
                                self.meta_data['crop_width'],
                                self.meta_data['label_type'])
        return prototxt_str


    def solver(self):
        solverprototxt = tools.CaffeSolver(trainnet_prototxt_path=osp.join(self.model_dir, "trainnet.prototxt"),
                                           testnet_prototxt_path=osp.join(self.model_dir, "valnet.prototxt"))
        solverprototxt.sp['solver_mode'] = self.meta_data['solver_mode']

        solverprototxt.sp['display'] = str(self.meta_data['caffe_display'])
        solverprototxt.sp['base_lr'] = str(self.meta_data['base_lr'])
        solverprototxt.sp['max_iter'] = str(self.meta_data['caffe_max_iter'])
        solverprototxt.sp['momentum'] = str(self.meta_data['momentum'])
        solverprototxt.sp['weight_decay'] = str(self.meta_data['caffe_weight_decay'])
        solverprototxt.sp['stepsize'] = str(self.meta_data['caffe_stepsize'])
        solverprototxt.sp['gamma'] = str(self.meta_data['caffe_gamma'])
        solverprototxt.sp['lr_policy'] = '\"' + self.meta_data['caffe_lr_policy'] + '\"'
        solverprototxt.sp['regularization_type'] = '\"' + self.meta_data['regularization_type'] + '\"'
        solverprototxt.sp['snapshot_prefix'] = '\"' + self.snapshot_dir + '\"'

        solverprototxt.sp['snapshot'] = '1000000'  # dummmy value
        solverprototxt.sp['test_interval'] = '1000000'  # dummmy value
        solverprototxt.sp['test_iter'] = '1000000000'  # dummmy value as we handle our own validation

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

    def initialize_solver(self):
        print("initializing solver...")
        solver_type = self.meta_data['solver']
        if solver_type == 'SGD':
            solver = caffe.SGDSolver(osp.join(self.model_dir, 'solver.prototxt'))
            solver.test_nets[0].share_with(solver.net)
            return solver
        elif solver_type == 'adam':
            solver = caffe.SGDSolver(osp.join(self.model_dir, 'solver.prototxt'))
            solver.test_nets[0].share_with(solver.net)
            return solver
        else:
            print('Other solver types are not implemented. However, they are supposed to have the same signature')

    def set_solver(self, snapshot_weight=None,
                   snapshot_state=None, snapshot_history=None):
        self.model()
        self.solver = self.initialize_solver()
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


