from utilities.tools import *
from nets.caffe_demo_net import DemoNetModel

if __name__ == "__main__":

    download_dataset("http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz", "cifar10.tar.gz")
    data = unpickle('data/cifar-10-batches-py/data_batch_2')


    data_folder = '/home/amir/echoProject/TMI/file_lists/'
    list_train = data_folder + 'list_train_demo.txt'
    list_test = data_folder + 'list_test_demo.txt'
    model = DemoNetModel()
    model.set_data(train_list_file=list_train, valid_list_file=list_test)
    model.set_solver()
    validation_accuracy = model.train_validate()

