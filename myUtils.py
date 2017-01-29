import matplotlib.pyplot as plt
import constants as c
import glob
import numpy as np

def plot_show(data, block=False):
    l1 = plt.plot(data[:, c.TRAIN_LOSS], 'b')
    l2 = plt.plot(data[:, c.TRAIN_ACCURACY], 'g')
    l3 = plt.plot(data[:, c.VAL_LOSS], 'b--')
    l4 = plt.plot(data[:, c.VAL_ACCURACY], 'g--')
    plt.legend(['Train_loss', 'Train_Acc', 'Valid_Loss', 'Valid_Acc'])
    # plt.legend(['Train_Acc', 'Valid_Acc'])
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
            plot_show(t, True)


