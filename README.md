# Deep Batch: Deep Learning python wrapper for data handling [Caffe and Keras]

Our main motivation for developing the framework, is to deal with datasets where training data is not uniformly distributed among classes[values], both in generative and discriminative models. To mitigate this problem, we have developed a DataHanlder which holds label maps of the training set and generates mini-batches, on-the-fly, which holds equal (semi-equal) number of samples from each class[value].

The given framework provides the following functionalities:
- Reading data using a list_file index
- On-the-fly batch selection via different strategies of:
 - uniform
 - random
 - iterative
- On-the-fly data augmentation including random rotation and translation
- Input data handling for
 - sequential data,
 - multi-stream networks with multiple input and multiple outputs
- Maintaining training history and log

## Project Hierarchy
The framework is implemented based on the following class diagram:

    RootModel                  # Abstract class  [shared functionalities between Caffe and Keras]
    ├── RootCaffeModel         # Abstract class  [Caffe-specific functionality]
        ├── MyCaffeNet         # A sample Caffe model which implements abstract methods
    ├── RootKerasModel         # Abstract class  [Keras-specific functionality]
        ├── MyKerasNet         # A sample Keras model which implements abstract methods

Data handling are all handled in the DataHandler class (utilities/datahandler.py). These functionalities include, but are not limited to:
- On-the-fly data preprocessing and augmentation
 - resize and crop
 - random rotation from uniform or normal distributions
 - random translation from uniform or normal distributions
- On-the-fly uniform/random/iterative batch selection
- Read image (via opencv) and MATLAB (via scipy) datasets 

Datahandler expects the data to be provided in list files. Each list files should follow this design:
    
    fileaddress, label_0[,label_1, label_2, ...]
    fileaddress, label_0[,label_1, label_2, ...]
    fileaddress, label_0[,label_1, label_2, ...]
    ...
As demonstrated above, each sample can have multiple labels. However, one needs to decide the label used for training via the `main_label_index` hyper-parameter.

## How to use
To take advantage of functionalities implemented in the framework, you need to write your own class which inherits from the RootCaffeModel or RootKerasModel, based on your preference. Your class needs to override the following methods:
    
    init_meta_data      # defines all the hyperparameters for training, datahandler, optimizer, etc.
    net                 # define the network architecture
    train_validate      # main training method which holds the training and validation loops
    evaluate[optional]  # to evaluate performance of a trained model on a given test set 
    [Other methods can be implemented based on necessity, such as output-specific accuracy calculator]

The following two sample classes are implemented to demonstrate functionalities of the framework and provide guidance:
    
    net
    ├── caffe_demo_net
        ├── CaffeDemoClassificationNet
    ├── keras_demo_net
        ├── KerasDemoMultiStreamRegressionNet
        
The CaffeDemoClassificationNet is a simple convolutional caffe model, which will be trained on the [Cifar10 dataset] (https://www.cs.toronto.edu/~kriz/cifar.html) if you run `main_caffe_demonet.py`.

The KerasDemoMultiStreamRegressionNet is a sample MultiStream, Sequential, Keras model, which is **hypothetically** trained on the [Cifar10 dataset] (https://www.cs.toronto.edu/~kriz/cifar.html). Please consider that the Cifar10 dataset is not a multi-input, multi-output dataset, however in the `main_keras_demonet.py`, we have devided the training and validation set into 4 sub-sets, **assuming** that each represent a different input type. The multi-stream network, **designed only for demonstration purposes**, has a shared architecture and shares the weights among the first few layers, while each stream has its own stream-specific layers. The model end with recurrent LSTM layers; however, since Cifar10 is not a sequential dataset, we faked that by considering the three channels of the color images as three different frames of a sequencial video. **KerasDemoMultiStreamRegressionNet is implemented only to demonstrate capabilities of the Deep Batch framework and the trained model shall not be trusted**.

### Library dependencies
Deep Batch depends on the following libraries:
- caffe
- keras
- tensorflow
- cv2
- numpy
- scipy
- sklearn
- matplotlib
- json

### Limitations
- So far, we have only developed for the TensorFlow backend of Keras.
- Data handler accepts the data only in `list index files` and does not directly read contents of given directories
- Label types, other than single value numbers (such as mask-images), are not handled.
