from RootKerasModel import RootKerasModel
import numpy as np

if __name__ == "__main__":

    list_files_folder = '/home/amir/echoData/7views_cine/list_files/'
    # list_files_folder  = '/home/amir/echoProject/TMI/file_lists/'
    list_train_str = list_files_folder + 'list_train'
    list_valid_str = list_files_folder + 'list_valid'


    list_train = []
    list_valid = []

    # list_train.append(list_train_str)
    # list_valid.append(list_valid_str)

    selected_views = range(0, 7)
    for i in selected_views:
        list_train.append(list_train_str + str(i))
        list_valid.append(list_valid_str + str(i))

    range_views = np.array([8, 7, 10, 12, 4, 7, 5])
    external_dict = {'range_views': range_views[list(selected_views)]}

    model = RootKerasModel(external_dict)
    model.set_data(train_list_file=list_train, valid_list_file=list_valid)
    model.set_solver()
    validation_accuracy = model.train_validate()