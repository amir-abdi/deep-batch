from RootKerasModel import RootKerasModel
import numpy as np
import sys
sys.path.append('/home/amir/keras/keras/layers')
sys.path.append('/home/amir/keras/keras/build/lib/layers')


# ------------------- load data list files ---------------------------
list_files_folder = '/home/amir/echoData/7views_cine/list_files/'
# list_files_folder  = '/home/amir/echoProject/TMI/file_lists/'
list_train_str = list_files_folder + 'list_train'
list_valid_str = list_files_folder + 'list_valid'
list_test_str = list_files_folder + 'list_test'

views = ['AP2', 'AP3', 'AP4', 'PLAX', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)']

list_train = []
list_valid = []
list_test = []

# list_train.append(list_train_str)
# list_valid.append(list_valid_str)

selected_views = [0, 1, 2, 3, 4, 6]  # range(0, 7) #  removed PLAX and PSAX(M)  - 3 and 5
for i in selected_views:
    list_train.append(list_train_str + str(i))
    list_valid.append(list_valid_str + str(i))
    list_test.append(list_test_str + str(i))

range_views = np.array([8, 7, 10, 12, 4, 7, 5])
external_dict = {'range_views': range_views[list(selected_views)]}
# --------------------------------------------------------------------

if __name__ == "__main__":
    model = RootKerasModel(external_dict)
    model.set_data(train_list_file=list_train, valid_list_file=list_valid)
    model.set_solver()
    validation_accuracy = model.train_validate()

if __name__ == "__main__test": # test
    model = RootKerasModel(external_dict)
    weight_file = 'models/RootKerasModel/snapshots/train_2017.2.17_6views_fixedValid'
    model.set_data(test_list_file=list_test)
    validation_accuracy = model.evaluate(weight_file)