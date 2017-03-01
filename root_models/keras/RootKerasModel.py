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

    def write_snapshot(self, solver, type_str):
        file = str(self.write_filename + type_str + '.kerasmodel')
        solver.save(file)
        print('Wrote snapshot to: {:s}'.format(file))
