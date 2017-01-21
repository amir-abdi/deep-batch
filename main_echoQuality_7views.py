from nets.echo_quality_7 import EchoNet7

model_counter = 1

if __name__ == "__main__":
    data_folder = '/media/truecrypt1/610cases_echoQuality_2016.11.18/2016.11.25/MatAnon/'
    list_train = data_folder + 'list_train_demo.txt'
    list_test = data_folder + 'list_test_demo.txt'

    model = EchoNet7()
    model.set_data(train_list_file=list_train, valid_list_file=list_test)
    model.set_solver()
    validation_accuracy = model.train_validate()

