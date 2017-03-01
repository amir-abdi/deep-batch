# Deep Batch: Deep Learning python wrapper for data handling [Caffe and Keras]

The given framework handles the following:
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

## How to use
To take advantage of functionalities implemented in the framework, you need to write your own class which inherits from the RootCaffeModel or RootKerasModel, based on your preference. Your class needs to override the following methods:
    
    init_meta_data      # defines all the hyperparameters for training, datahandler, optimizer, etc.
    net                 # define the network architecture
    train_validate      # main training method which holds the training and validation loops
    evaluate[optional]  # to evaluate performance of a trained model on a given test set 
Other methods can be implemented based on necessity.

The following two sample classes are implemented to demonstrate functionalities of the framework and provide guidance:
    
    net
    ├── caffe_demo_net
        ├── CaffeDemoClassificationNet
    ├── keras_demo_net
        ├── KerasDemoMultiStreamRegressionNet
        
