from nets.caffe_echo_quality_7 import EchoNet7

model_counter = 1

if __name__ == "__main__":
    list_files_folder = '/media/truecrypt1/610cases_echoQuality_2016.11.18/2016.11.25/list_files/'
    list_train_str = list_files_folder + 'list_train'
    list_valid_str = list_files_folder + 'list_test'

    list_train = []
    list_valid = []
    for i in range(1):
        list_train.append(list_train_str + str(i))
        list_valid.append(list_valid_str + str(i))

    model = EchoNet7()
    model.set_data(train_list_file=list_train, valid_list_file=list_valid)
    model.set_solver()
    validation_accuracy = model.train_validate()

