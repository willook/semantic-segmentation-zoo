import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Builds the conv block for UNets
    Apply successivly a 2D convolution, BatchNormalization relu
    """
    net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=kernel_size, stride=[2, 2], activation_fn=None)
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)
    return net

def build_unet(inputs, preset_model, num_classes):

    #####################
    # Downsampling path #
    #####################

    net = ConvBlock(inputs, 64)
    net = ConvBlock(net, 64)
    skip_1 = net

    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    net = ConvBlock(net, 128)
    net = ConvBlock(net, 128)
    skip_2 = net

    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    net = ConvBlock(net, 256)
    net = ConvBlock(net, 256)
    skip_3 = net

    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    net = ConvBlock(net, 512)
    net = ConvBlock(net, 512)
    skip_4 = net

    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
    net = ConvBlock(net, 1024)
    net = ConvBlock(net, 1024)

    ####################
    # Upsampling path  #
    ####################

    net = conv_transpose_block(net, 512)
    net = tf.add(net, skip_4)
    net = ConvBlock(net, 512)
    net = ConvBlock(net, 512)

    net = conv_transpose_block(net, 256)
    net = tf.add(net, skip_3)
    net = ConvBlock(net, 256)
    net = ConvBlock(net, 256)

    net = conv_transpose_block(net, 128)
    net = tf.add(net, skip_2)
    net = ConvBlock(net, 128)
    net = ConvBlock(net, 128)

    net = conv_transpose_block(net, 64)
    net = tf.add(net, skip_1)
    net = ConvBlock(net, 64)
    net = ConvBlock(net, 64)

    #####################
    #      Softmax      #
    #####################
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net
