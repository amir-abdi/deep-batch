from abc import ABCMeta

import numpy as np

from root_models.RootModel import RootModel


class RootKerasModel(RootModel):
    __metaclass__ = ABCMeta

    def __init__(self, external_dict=None):
        super().__init__(external_dict)

    def model(self):
        self.net()

    def get_solver(self):
        return self.net_model

    def calculate_accuracy_from_absErr(self, abs_errs):
        abs_errs = np.asarray(abs_errs)
        if self.meta_data['scale_label'] == 0:
            range_views = self.meta_data['range_views']
        else:
            range_views = np.ones(abs_errs.shape)

        metric = np.mean((range_views - abs_errs) / range_views)
        return metric

    def write_snapshot(self, solver, type_str):
        file = str(self.write_filename + type_str + '.kerasmodel')
        solver.save(file)
        print('Wrote snapshot to: {:s}'.format(file))
