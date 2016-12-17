import socket
if socket.gethostname() == 'purang27':
    caffe_root = '/home/amir/caffe/'
    framework_root = '/home/amir/echoProject/framework/'

if __name__ == "__main__":
    print("host name:", socket.gethostname())