from utilities.tools import *
from nets.caffe_demo_net import CaffeDemoClassificationNet
from keras.datasets import cifar10


if __name__ == "__main__":
    data = [[None], [None], [None], [None]]
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
    data[0][0] = X_train
    data[1][0] = y_train
    data[2][0] = X_valid
    data[3][0] = y_valid

    model = CaffeDemoClassificationNet()
    model.set_trainvalid_data(data=data)
    model.set_solver()
    validation_accuracy = model.train_validate()

