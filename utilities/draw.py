# import matplotlib
# matplotlib.use('Agg')
import glob

import matplotlib.pyplot as plt
import numpy as np

from utilities import constants as c


def plot_training_history(data, block=False, num_streams=0):
    plt.close()

    # l1 = plt.plot(data[:, c.TRAIN_LOSS], 'b', label='Train_loss'))
    l2 = plt.plot(data[1:, c.TRAIN_ACCURACY], 'k', label='TrainAcc')
    # l3 = plt.plot(data[:, c.VAL_LOSS], 'b--', label='Valid_Loss'))
    l4 = plt.plot(data[1:, c.VAL_ACCURACY], 'k--', label='ValidAcc')

    if num_streams > 1:
        for stream in range(num_streams):
            plt.plot(data[1:, 4+stream], label='view {}'.format(str(stream)))

    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.show(block=block)
    plt.pause(0.001)


def read_json_block(file):
    temp = ''
    while True:
        line = file.next().strip()
        temp += line
        if '}' in line:
            break
    return temp


if __name__ == "__main__":
    fld = 'the folder which holds training_histories'
    files = glob.glob(fld+ '/*.npy')

    filter = '1,1'
    for file in files:
        print(file)
        if filter in file:
            t = np.load(file)
            plot_training_history(t, True)


