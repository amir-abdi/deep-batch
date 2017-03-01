from nets.keras_demo_net import KerasDemoMultiStreamRegressionNet
from keras.datasets import cifar10
import numpy as np



if __name__ == "__main__":
    # This is the cifar10 dataset being treated as a hypothetical sequential dataset with 3 frames per each sample.
    # The RGB channels of cifar images are being recognized as consecutive greyscale frames

    data = [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
    total = X_train.shape[0] / 1000
    for stream in range(4):
        lower = int(np.floor(stream * total / 4))
        upper = int(np.floor((stream + 1) * total / 4))
        data[0][stream] = X_train[lower:upper]
        data[1][stream] = y_train[lower:upper]
        data[2][stream] = X_valid[lower:upper]
        data[3][stream] = y_valid[lower:upper]

    model = KerasDemoMultiStreamRegressionNet()
    model.set_trainvalid_data(data=data)
    model.set_solver()
    validation_accuracy = model.train_validate()
