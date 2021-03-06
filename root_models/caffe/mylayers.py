from caffe import layers
from caffe import params


def conv_relu(bottom, numout, ks=3, stride=1, pad=0, same_size=False):
    if same_size:
        pad = int(ks / 2)
    conv = layers.Convolution(bottom, kernel_size=ks, stride=stride,
                              num_output=numout, pad=pad,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    return conv, layers.ReLU(conv, in_place=True)


def conv(bottom, numout, ks=3, stride=1, pad=0, same_size=False):
    if same_size:
        pad = int(ks / 2)
    conv = layers.Convolution(bottom, kernel_size=ks, stride=stride,
                              num_output=numout, pad=pad,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    return conv


def deconv_relu(bottom, numout, ks=3, stride=1, pad=0):
    deconv = layers.Deconvolution(bottom,
                                  convolution_param=dict(kernel_size=ks, stride=stride, num_output=numout, pad=pad,
                                                    weight_filler=dict(type='gaussian', std=0.01)),
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return deconv, layers.ReLU(deconv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return layers.Pooling(bottom, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)


def fc_relu(bottom, nout):
    fc = layers.InnerProduct(bottom, num_output=nout, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1),
                             param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)])
    return fc, layers.ReLU(fc, in_place=True)



def fc(bottom, nout, bias_constant=1):
    fc = layers.InnerProduct(bottom, num_output=nout, weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=bias_constant),
                             param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)])
    return fc


def data_label(proto, batch_size, channels, im_height, im_width, label_type):
    data = ("input: \"data\"\n" \
             "input_shape {\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "}\n\n" % (
                 batch_size, channels, im_height, im_width))
    if label_type == 'single_value':
            label = ("input: \"label\"\n" \
                 "input_shape {\n" \
                 "dim: %d\n" \
                 "}\n\n\n" % (
                     batch_size))
    elif label_type == 'mask_image':
            label = ("input: \"label\"\n" \
                 "input_shape {\n" \
                 "dim: %d\n" \
                 "dim: %d\n" \
                 "dim: %d\n" \
                 "dim: %d\n" \
                 "}\n" % (batch_size, channels, im_height, im_width))

    return data + label +"\n".join(proto.split("\n")[12:])

def data(proto, batch_size, channels, im_height, im_width, label_type):
    data = ("input: \"data\"\n" \
             "input_shape {\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "dim: %d\n" \
             "}\n\n" % (
                 batch_size, channels, im_height, im_width))

    return data +"\n".join(proto.split("\n")[12:])