#import caffe nets
from nets.caffe_echo_net_x import EchoXNetModel

if __name__ == "__main__":
    data_folder = '/home/amir/echoProject/TMI/file_lists/'
    list_train = data_folder + 'list_train.txt'
    list_test = data_folder + 'list_test.txt'

    hyper_meta_data = {'model_variant': 'echo_netx',
                       'cnn_layers': 'three',
                       'kernel_num1': int(22),
                       'kernel_size1': 17,
                       'kernel_num2': int(53),
                       'kernel_size2': 11,
                       'kernel_num3': int(64),
                       'kernel_size3': 11,
                       'fc1': 1079,
                       'fc2': 699,
                       'model_counter': '0',  #5,6,7 --> 200 epochs
                       'cv_counter': '0',
                       'generation_counter': 0,
                       }

    model = EchoXNetModel(external_meta_data=hyper_meta_data)
    model.set_data(train_list_file=list_train, valid_list_file=list_test)
    print 'metadata: ', model.meta_data
    model.set_solver()
    validation_accuracy = model.train_validate()
                                               #  snapshot_weight=snap_fld+'0,0_last.caffemodel',
                                               # snapshot_history=snap_fld+'0,0_history.npy',
                                               # snapshot_state=snap_fld+'0,0.state')



