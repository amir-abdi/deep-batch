#import caffe nets
from nets.demo_net import DemoNetModel

model_counter = 1

if __name__ == "__main__":
    data_folder = '/home/amir/echoProject/TMI/file_lists/'
    list_train = data_folder + 'list_train_demo.txt'
    list_test = data_folder + 'list_test_demo.txt'

    model = DemoNetModel()
    model.set_data(train_list_file=list_train, valid_list_file=list_test)
    model.set_solver()
    validation_accuracy = model.train_validate()

