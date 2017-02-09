from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne
import pdb
import scipy.io as scio

from scipy.misc import imresize
from cv2 import imread
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import metrics
import matplotlib.pyplot as plt

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)
np.random.seed(42)


def load_data():
    trDataset = scio.loadmat('Cardiac_train_dataset.mat', squeeze_me=True,
                             struct_as_record=False)
    tsDataset = scio.loadmat('Cardiac_test_dataset.mat', squeeze_me=True,
                             struct_as_record=False)

    train_images = trDataset['images']
    train_labels = trDataset['labels']
    train_labels = np.abs(train_labels)

    test_images = tsDataset['images']
    test_labels = tsDataset['labels']
    test_labels = np.abs(test_labels)

    # normalize the images in terms of mean and standard deviation
    try:
        with np.load('cardiacStatsNoAugment_120.npz') as mammoStats:
            mean_images = mammoStats['mean_images']
            std_images = mammoStats['std_images']

    except:
        mean_images, std_images = getImageStats(train_images)
        np.savez('cardiacStatsNoAugment_120.npz',
                 mean_images=mean_images,
                 std_images=std_images,
                 )

    return dict(
        train_images=train_images,
        train_labels=train_labels.astype('float32'),
        test_images=test_images,
        test_labels=test_labels.astype('float32'),
        mean_images=mean_images,
        std_images=std_images)


def getImageStats(images):
    bs = 30  # batch size
    stdData = []
    for t in range(0, len(images), bs):
        batch = images[t:min((t + bs, len(images)))]
        # pdb.set_trace()
        print('collecting image stats: batch starting with image {}\n'.format(batch[0]))
        tmp = getBatch(batch)
        data = tmp
        meanBatch = np.mean(data, axis=0)
        if t == 0:
            meanData = meanBatch
        else:
            meanData = np.mean((meanData, meanBatch), axis=0)

    return meanData, stdData


def getBatch(images, *args):
    if len(args) == 1:
        meanData = args[0]
        meanData = np.float32(meanData)
    else:
        meanData = []

    imo = []
    for cine_images in images:
        imt = np.float32(imread(cine_images))
        imt = imresize(imt[:, :, 1], (120, 120), interp='cubic')
        if meanData:
            norData = imt
            norData -= meanData
            # norData/= stdData
        else:
            norData = imt
        imo.append(norData)
    imageData = np.asarray(imo)
    return np.float32(imageData)


# ##################### Build the neural network model #######################

from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers import ConcatLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import LSTMLayer
from lasagne.layers import MaxPool2DLayer
import theano.tensor as T

from lasagne.utils import as_theano_expression


def build_cnn(input_var_cc=None, n=1):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = ConvLayer(NonlinearityLayer(batch_norm(l), nonlinearity=rectify), num_filters=out_num_filters,
                            filter_size=(3, 3), stride=first_stride, nonlinearity=None,
                            pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

        stack_2 = ConvLayer(NonlinearityLayer(batch_norm(stack_1), nonlinearity=rectify), num_filters=out_num_filters,
                            filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                            pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = ConvLayer(NonlinearityLayer(batch_norm(l), nonlinearity=rectify),
                                       num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                       nonlinearity=None,
                                       pad='same', b=None, flip_filters=False)
                block = ElemwiseSumLayer([stack_2, projection])
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = ElemwiseSumLayer([stack_2, padding])
        else:
            block = ElemwiseSumLayer([stack_2, l])

        return block

    # Building the sub network for CC image
    l_in_cc_image = InputLayer(shape=(None, 1, 120, 120), input_var=input_var_cc)

    # first layer, output is 16 x 32 x 32 for the CC sub net
    l_cc_image = ConvLayer(l_in_cc_image, num_filters=8, filter_size=(3, 3), stride=(2, 2),
                           nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    # first stack of residual blocks, output is 16 x 32 x 32re
    for _ in range(n):
        l_cc_image = residual_block(l_cc_image)

    # second stack of residual blocks, output is 32 x 16 x 16 for CC and ML sub net
    l_cc_image = residual_block(l_cc_image, increase_dim=True)

    for _ in range(1, n):
        l_cc_image = residual_block(l_cc_image)

    # third stack of residual blocks, output is 64 x 8 x 8
    l_cc_image = residual_block(l_cc_image, increase_dim=True)

    for _ in range(1, n):
        l_cc_image = residual_block(l_cc_image)


    #lstm layer for getting the temporal featrues
    l_gpooling = GlobalPoolLayer(l_cc_image)
    l_reshape = ReshapeLayer(l_gpooling, (-1, 30, [1]))
    l_lstm1 = LSTMLayer(l_reshape, num_units=30, grad_clipping=1)
    l_lstm2 = LSTMLayer(l_lstm1, num_units=30, grad_clipping=1)
    l_shp = ReshapeLayer(l_lstm2, (-1, 30))

    # fully connected layer
    network = DenseLayer(l_shp, num_units=1,W=lasagne.init.HeNormal())
    return network


def align_targets(predictions, targets):
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
                getattr(targets, 'ndim', None) == 1):
        targets = as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets


def struct_loss(predictions, targets):
    # defining the struct loss
    alpha = 0.1#change this to assign the weight to the structured loss function
    predictions, targets = align_targets(predictions, targets)
    loss_inc = (T.gt(targets[1:], targets[0:-1])) * (T.maximum(0, predictions[0:-1] - predictions[1:]))
    loss_dec = (T.lt(targets[1:], targets[0:-1])) * (T.maximum(0, predictions[1:] - predictions[0:-1]))
    loss_temp = 0.5 * (loss_inc.mean() + loss_dec.mean())

    # define the mean square loss
    loss_mse = lasagne.objectives.squared_error(predictions, targets)

    # weighting both the loss
    total_loss = alpha * loss_temp + (1-alpha) * loss_mse.mean()
    return total_loss


def compute_caption_error_old(y_hat_frames, y_true_frames, frame_len):
    # reshape the
    num_batches = y_hat_frames.shape[0] / frame_len
    y_hat_frames = np.reshape(y_hat_frames, (num_batches, frame_len))
    y_true_frames = np.reshape(y_true_frames, (num_batches, frame_len))

    y_hat_sys = np.argmin(y_hat_frames, axis=1)
    y_true_sys = np.argmin(y_true_frames, axis=1)

    y_hat_dias = np.argmax(y_hat_frames[0:y_hat_sys], axis=1)
    y_true_dias = np.argmax(y_true_frames[0:y_true_sys], axis=1)

    sys_error = np.abs(y_true_sys - y_hat_sys)
    dias_error = np.abs(y_true_dias - y_hat_dias)

    return sys_error.mean(), dias_error.mean()

    ###############################
def compute_caption_error(y_hat_frames, y_true_frames, frame_len):
        # reshape the
    num_batches = y_hat_frames.shape[0] / frame_len
    y_hat_frames = np.reshape(y_hat_frames, (num_batches, frame_len))
    y_true_frames = np.reshape(y_true_frames, (num_batches, frame_len))

    y_hat_sys = np.argmin(y_hat_frames, axis=1)
    y_true_sys = np.argmin(y_true_frames, axis=1)

    indices_y_hat_frames = np.arange(y_hat_sys.shape[0])

    y_hat_dias_list = [y_hat_frames[indices_y_hat_frames[num], 0:y_hat_sys[num] + 1]
                  for num in range(len(y_hat_sys))]

    indices_y_true_frames = np.arange(y_true_frames.shape[0])
    y_true_dias_list = [y_true_frames[indices_y_true_frames[num], 0:y_true_sys[num] + 1]
                   for num in range(len(y_true_sys))]

    y_hat_dias = [np.argmax(tmp_y_hat_dias) for tmp_y_hat_dias in y_hat_dias_list]
    y_true_dias = [np.argmax(tmp_y_true_dias) for tmp_y_true_dias in y_true_dias_list]

    y_hat_dias = np.squeeze(y_hat_dias)
    y_true_dias = np.squeeze(y_true_dias)

    sys_error = np.abs(y_true_sys - y_hat_sys)
    dias_error = np.abs(y_true_dias - y_hat_dias)

    return sys_error.mean(), dias_error.mean()

def iterate_minibatches(images, targets, mean_images, batchsize):
    indices = np.arange(len(targets))
    for start_idx in range(0, len(images) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        batch_data = np.zeros((batchsize, 1, 120, 120))
        # pdb.set_trace()
        for cnt, excerpt_indices in enumerate(excerpt):
            image_data = np.float32(imread(images[excerpt_indices]))
            image_data = imresize(image_data[:, :, 1], (120, 120), interp='cubic')
            image_data = np.float32(image_data)
            image_data -= mean_images
            # image_data /= std_images
            batch_data[cnt, :, :, :] = np.float32(image_data)

        yield lasagne.utils.floatX(batch_data), targets[excerpt]


# ############################## Main program ################################

def main(n=5, num_epochs=500, batch_size=90, model=None):
    print("Loading data...")
    data = load_data()

    # assign the folder for storing the temporary data and results from training
    if model is None:
        result_folder = 'model_resnet_struct_'+str(n)
        fig_folder = 'fig_resnet_struct_'+str(n)
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        if not os.path.isdir(fig_folder):
            os.mkdir(fig_folder)

    # prepare the training data
    train_images = data['train_images']

    # load the stat of the dataset
    mean_images = data['mean_images']
    std_images = data['std_images']

    # Prepare the test data
    test_images = data['test_images']

    # Prepate the training and test annotations
    Y_train = data['train_labels']
    Y_test = data['test_labels']

    # Prepare Theano variables for inputs and targets
    input_var_cc = T.tensor4('inputs')
    target_var = T.vector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var_cc, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = struct_loss(prediction, target_var)
    # loss = theano.function([prediction, target_var], struct_loss(prediction, target_var))
    # loss = lasagne.objectives.squared_error(prediction, target_var)
    # loss = loss.mean()
    # add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.001

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.001  # for skips 1, lr = 0.005; for depth_5, lr =0.0005
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var_cc, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var_cc, target_var], [loss, test_prediction])

    if model is None:
        # launch the training loop
        print("Starting training...")
        train_loss_epoch = np.array([])
        train_abs_error_epoch = np.array([])
        train_r2_error_epoch = np.array([])
        train_mse_error_epoch = np.array([])
        train_sys_error_epoch = np.array([])
        train_dys_error_epoch = np.array([])

        val_loss_epoch = np.array([])
        val_abs_error_epoch = np.array([])
        val_r2_error_epoch = np.array([])
        val_mse_error_epoch = np.array([])
        val_sys_error_epoch = np.array([])
        val_dys_error_epoch = np.array([])

        # We iterate over epochs:
        for epoch in range(num_epochs):
            train_indices = np.arange(len(Y_train))
            # np.random.shuffle(train_indices)
            train_images = train_images[train_indices]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_batches = 0
            start_time = time.time()
            batch_cnt = 0
            train_targets = []
            train_predid = []
            train_loss = []

            for batch in iterate_minibatches(train_images,
                                             Y_train,
                                             mean_images,
                                             batch_size):
                inputs, targets = batch
                num_batch, num_dim, row, col = inputs.shape

                inputs_images = np.reshape(inputs[:, 0, :, :], (num_batch, 1, row, col))

                loss = train_fn(inputs_images, targets)

                _, scores = val_fn(inputs_images, targets)
                predid = np.squeeze(scores)
                train_batches += 1
                train_loss = np.concatenate([train_loss, [loss]])
                train_targets = np.concatenate([train_targets, targets])
                train_predid = np.concatenate([train_predid, predid])
                print('Training Batch {:.0f} of Epoch {:.0f}/{:.0f}\n'.format(batch_cnt, epoch, num_epochs))
                batch_cnt += 1

            # save the model
            model_name = result_folder + '/' + 'resnet__model_epoch_' + str(epoch) + '.npz'
            np.savez(model_name, *lasagne.layers.get_all_param_values(network))

            train_abs_error = metrics.mean_absolute_error(train_targets, train_predid)
            train_r2_score = metrics.r2_score(train_targets, train_predid)
            if train_r2_score < 0:
                train_r2_score = -1
            train_mean_error = metrics.mean_squared_error(train_targets, train_predid)
            train_sys_error, train_dys_error = compute_caption_error(train_predid, train_targets, 30)
            mean_train_loss = np.mean(train_loss)
            # And a full pass over the validation data:
            # pdb.set_trace()
            val_batches = 0
            batch_cnt = 0
            val_targets = []
            val_predid = []
            val_loss = []

            val_batch_size = batch_size
            for batch in iterate_minibatches(test_images,
                                             Y_test,
                                             mean_images,
                                             val_batch_size):
                inputs, targets = batch
                num_batch, num_dim, row, col = inputs.shape

                inputs_images = np.reshape(inputs[:, 0, :, :], (num_batch, 1, row, col))
                #pdb.set_trace()

                loss, scores = val_fn(inputs_images, targets)
                predid = np.squeeze(scores)
                val_batches += 1
                val_loss = np.concatenate([val_loss, [loss]])
                val_predid = np.concatenate([val_predid, predid])
                val_targets = np.concatenate([val_targets, targets])
            val_abs_error = metrics.mean_absolute_error(val_targets, val_predid)
            val_r2_score = metrics.r2_score(val_targets, val_predid)
            val_mean_error = metrics.mean_squared_error(val_targets, val_predid)
            val_sys_error, val_dys_error = compute_caption_error(val_predid, val_targets, 30)
            mean_val_loss = np.mean(val_loss)

            if train_abs_error > 1:
                train_abs_error = 1
            if train_mean_error > 1:
                train_mean_error = 1
            if train_r2_score < 0:
                train_r2_score = 0
            if val_abs_error > 1:
                val_abs_error = 1
            if val_mean_error > 1:
                val_mean_error = 1
            if val_r2_score < 0:
                val_r2_score = 0

                # Then we print the results for this epoch:
            train_abs_error_epoch = np.concatenate([train_abs_error_epoch, [train_abs_error]])
            train_r2_error_epoch = np.concatenate([train_r2_error_epoch, [train_r2_score]])
            train_mse_error_epoch = np.concatenate([train_mse_error_epoch, [train_mean_error]])
            train_dys_error_epoch = np.concatenate([train_dys_error_epoch, [train_dys_error]])
            train_sys_error_epoch = np.concatenate([train_sys_error_epoch, [train_sys_error]])
            train_loss_epoch = np.concatenate([train_loss_epoch, [mean_train_loss]])

            val_abs_error_epoch = np.concatenate([val_abs_error_epoch, [val_abs_error]])
            val_r2_error_epoch = np.concatenate([val_r2_error_epoch, [val_r2_score]])
            val_mse_error_epoch = np.concatenate([val_mse_error_epoch, [val_mean_error]])
            val_sys_error_epoch = np.concatenate([val_sys_error_epoch, [val_sys_error]])
            val_dys_error_epoch = np.concatenate([val_dys_error_epoch, [val_dys_error]])
            val_loss_epoch = np.concatenate([val_loss_epoch, [mean_val_loss]])

            # Draw the result of the experiment
            fig = Figure(figsize=(8, 8))
            fig.clear()
            canvas = FigureCanvas(fig)
            loss_plot = fig.add_subplot(111)
            train_line, = loss_plot.plot(train_loss_epoch, 'b')
            val_line, = loss_plot.plot(val_loss_epoch, 'r')
            loss_plot.legend((train_line, val_line), ('Training', 'Validation'))
            loss_plot.title.set_text('')
            loss_plot.set_xlabel('Num. Epoch')
            loss_plot.set_ylabel('Struct Loss')
            # abs_plot.set_yscale('log')
            canvas_name = fig_folder + '/' + 'cnn_struct_loss.png'
            canvas.print_figure(canvas_name)

            fig = Figure(figsize=(8, 8))
            fig.clear()
            canvas = FigureCanvas(fig)
            abs_plot = fig.add_subplot(111)
            train_line, = abs_plot.plot(train_abs_error_epoch, 'b')
            val_line, = abs_plot.plot(val_abs_error_epoch, 'r')
            abs_plot.legend((train_line, val_line), ('Training', 'Validation'))
            abs_plot.title.set_text('')
            abs_plot.set_xlabel('Num. Epoch')
            abs_plot.set_ylabel('Absolute Error')
            # abs_plot.set_yscale('log')
            canvas_name = fig_folder + '/' + 'cnn_cardiac_abs.png'
            canvas.print_figure(canvas_name)

            fig.clear()
            r2_plot = fig.add_subplot(111)
            train_line, = r2_plot.plot(train_r2_error_epoch, 'b')
            val_line, = r2_plot.plot(val_r2_error_epoch, 'r')
            r2_plot.legend((train_line, val_line), ('Training', 'Validation'))
            r2_plot.title.set_text('')
            r2_plot.set_xlabel('Num. Epoch')
            r2_plot.set_ylabel('R-2 Ratio')
            canvas_name = fig_folder + '/' + 'cnn_cardiac_r2_error.png'
            canvas.print_figure(canvas_name)

            fig.clear()
            mse_plot = fig.add_subplot(111)
            train_line, = mse_plot.plot(train_mse_error_epoch, 'b')
            val_line, = mse_plot.plot(val_mse_error_epoch, 'r')
            mse_plot.legend((train_line, val_line), ('Training', 'Validation'))
            mse_plot.title.set_text('')
            mse_plot.set_xlabel('Num. Epoch')
            mse_plot.set_ylabel('Mean Squared Error')
            # mse_plot.set_yscale('log')
            canvas_name = fig_folder + '/' + 'cnn_cardiac_mse.png'
            canvas.print_figure(canvas_name)

            # save the results as dictionary file
            result_filename = fig_folder + '/' + 'cnn_results' + '.npz'
            np.savez(result_filename,
                     Train_abs_error=train_abs_error_epoch,
                     Val_abs_error=val_abs_error_epoch,
                     Train_mse_error=train_mse_error_epoch,
                     Val_mse_error=val_mse_error_epoch,
                     Train_r2_errror=train_r2_error_epoch,
                     Val_r2_error=val_r2_error_epoch,
                     Train_dys_error=train_dys_error_epoch,
                     Val_dys_error=val_dys_error_epoch,
                     Train_sys_error=train_sys_error_epoch,
                     Val_sys_error=val_sys_error_epoch
                     )

            # save the results into excel csv format
            txt_filename = fig_folder + '/' + 'resnet_results' + '.csv'

            total_epoch = np.arange(0, epoch + 1)
            result = np.vstack([total_epoch,
                                train_abs_error_epoch,
                                train_r2_error_epoch,
                                train_mse_error_epoch,
                                train_sys_error_epoch,
                                train_dys_error_epoch,
                                val_abs_error_epoch,
                                val_r2_error_epoch,
                                val_mse_error_epoch,
                                val_sys_error_epoch,
                                val_dys_error_epoch
                                ])

            result = np.transpose(result)
            np.savetxt(txt_filename, result, delimiter=',')

            # also preint hte result of the experiment
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training struct loss:\t\t{:.6f}".format(mean_train_loss))
            print("  training mse_error:\t\t{:.6f}".format(train_mean_error))
            print("  training r2_score:\t\t{:.6f}".format(train_r2_score))
            print("  training  systolic error:\t{:.6f}".format(train_sys_error))
            print("  training diastolic  error:\t{:.6f}".format(train_dys_error))

            print("  validation struct loss:\t{:.6f}".format(val_mean_error))
            print("  validation mse_error:\t\t{:.6f}".format(val_mean_error))
            print("  validation r2_score:\t\t{:.6f}".format(val_r2_score))
            print("  validation systolic error:\t{:.6f}".format(val_sys_error))
            print("  validation diastolic  error:\t{:.6f}".format(val_dys_error))
            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            # if (epoch + 1) == 10 or (epoch + 1) == 30:
            # new_lr = sh_lr.get_value() * 0.1
            # print("New LR:" + str(new_lr))
            # sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez('Cardiac_Caption_Resnet_Model.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        with np.load(model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    for indxData in range(0, 2, 1):

        if indxData == 0:
            images = train_images
            labels = Y_train

        if indxData == 1:
            images = test_images
            labels = Y_test

        final_batches = 0
        final_targets = []
        final_scores = []
        final_batch_size = 30
        for batch in iterate_minibatches(images,
                                         labels,
                                         mean_images,
                                         final_batch_size):
            inputs, targets = batch
            num_batch, num_dim, row, col = inputs.shape

            inputs_images = np.reshape(inputs[:, 0, :, :], (num_batch, 1, row, col))

            _, scores = val_fn(inputs_images, targets)
            predid = scores
            final_batches += 1
            final_scores = np.concatenate([final_scores, np.squeeze(predid)])
            final_targets = np.concatenate([final_targets, np.squeeze(targets)])

            #save the test results to the folder
            if indxData ==1:
                fig = Figure(figsize=(8, 8))
                fig.clear()
                canvas = FigureCanvas(fig)
                final_plot = fig.add_subplot(111)
                gt_line, = final_plot.plot(targets, 'b', marker='o')
                predid_line, = final_plot.plot(predid, 'g', marker='o')
                final_plot.legend((gt_line, predid_line), ('Ground Truth', 'Prediction'))
                final_plot.title.set_text('')
                final_plot.set_xlabel('Num Frames')
                final_plot.set_ylabel('Labels')
                visual_results_folder = 'visual_Results_'+str(n)
                if not os.path.isdir(visual_results_folder):
                    os.mkdir(visual_results_folder)
                canvas_name = visual_results_folder + '/' + 'Test_Case__'+str(final_batches)+'.png'
                canvas.print_figure(canvas_name)

        final_mse_err = metrics.mean_squared_error(final_targets, final_scores)
        final_r2_score = metrics.r2_score(final_targets, final_scores)
        final_sys_error, final_dys_error = compute_caption_error(final_scores, final_targets, 30)
        if indxData == 0:
            train_mse_error = final_mse_err
            train_r2_score = final_r2_score
            train_prediction = final_scores
            train_labels = final_targets
            train_sys_error = final_sys_error
            train_dys_error = final_dys_error
            print("****** Final Training Results ******")
            print("Mean Square Error:\t\t{:.3f}".format(train_mse_error))
            print("R-2 Score:\t\t\t{:.3f}".format(train_r2_score))
            print("Systolic Error:\t{:.3f}".format(train_sys_error))
            print("Diastolic  Error:\t{:.3f}".format(train_dys_error))

        if indxData == 1:
            test_mse_error = final_mse_err
            test_r2_score = final_r2_score
            test_prediction = final_scores
            test_labels = final_targets
            test_sys_error = final_sys_error
            test_dys_error = final_dys_error
            print("******* Final Training Results ******")
            print("Mean Square Error:\t\t{:.3f}".format(test_mse_error))
            print("R-2 Score:\t\t\t{:.3f}".format(test_r2_score))
            print("Systolic Error:\t{:.3f}".format(test_sys_error))
            print("Diastolic  Error:\t{:.3f}".format(test_dys_error))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep residual learning  Model with structured loss on Cardiac dataset for fram captioning prolem.")
        print("Network architecture using CNN.")
        print()
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv)>2:
            kwargs['model'] = sys.argv[2]
main(**kwargs)



