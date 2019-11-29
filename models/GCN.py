import tensorflow as tf
#from tensorflow.contrib import slim
from builders import frontend_builder
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic deconv block for GCN
    Apply Transposed Convolution for feature map upscaling
    """
    conv2d_transpose_layer = tf.keras.layers.Conv2DTranspose(
        n_filters, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=None)
    net = conv2d_transpose_layer(inputs)
    return net

def BoundaryRefinementBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Boundary Refinement Block for GCN
    """
    conv_layer1 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size, padding='same', activation=None)
    conv_layer2 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size, padding='same', activation=None)
    add_layer = tf.keras.layers.Add()

    net = conv_layer1(inputs)
    net = tf.keras.activations.relu(net)
    net = conv_layer2(net)
    net = add_layer([inputs, net])

    return net

def GlobalConvBlock(inputs, n_filters=21, size=3):
    """
    Global Conv Block for GCN
    """
    conv_layer1 = tf.keras.layers.Conv2D(n_filters, kernel_size=[size, 1], padding='same', activation=None)
    conv_layer2 = tf.keras.layers.Conv2D(n_filters, kernel_size=[1, size], padding='same', activation=None)
    conv_layer3 = tf.keras.layers.Conv2D(n_filters, kernel_size=[1, size], padding='same', activation=None)
    conv_layer4 = tf.keras.layers.Conv2D(n_filters, kernel_size=[size, 1], padding='same', activation=None)
    add_layer = tf.keras.layers.Add()

    net_1 = conv_layer1(inputs)
    net_1 = conv_layer2(net_1)

    net_2 = conv_layer3(inputs)
    net_2 = conv_layer4(net_2)

    net = add_layer([net_1, net_2])
    
    return net

def build_gcn(inputs, num_classes, preset_model='GCN', frontend="ResNet101", weight_decay=1e-5, is_training=True, upscaling_method="bilinear", pretrained_dir="models"):
    """
    Builds the GCN model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      GCN model
    """
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    
    
    add_layer1 = tf.keras.layers.Add()
    add_layer2 = tf.keras.layers.Add()
    add_layer3 = tf.keras.layers.Add()
    add_layer4 = tf.keras.layers.Add()

    res = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]

    down_5 = GlobalConvBlock(res[0], n_filters=21, size=3)
    down_5 = BoundaryRefinementBlock(down_5, n_filters=21, kernel_size=[3, 3])
    down_5 = ConvUpscaleBlock(down_5, n_filters=21, kernel_size=[3, 3], scale=2)

    down_4 = GlobalConvBlock(res[1], n_filters=21, size=3)
    down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    down_4 = add_layer1([down_4, down_5])
    down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    down_4 = ConvUpscaleBlock(down_4, n_filters=21, kernel_size=[3, 3], scale=2)

    down_3 = GlobalConvBlock(res[2], n_filters=21, size=3)
    down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    down_3 = add_layer2([down_3, down_4])
    down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    down_3 = ConvUpscaleBlock(down_3, n_filters=21, kernel_size=[3, 3], scale=2)

    down_2 = GlobalConvBlock(res[3], n_filters=21, size=3)
    down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    down_2 = add_layer3([down_2, down_3])
    down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    down_2 = ConvUpscaleBlock(down_2, n_filters=21, kernel_size=[3, 3], scale=2)

    net = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    net = ConvUpscaleBlock(net, n_filters=21, kernel_size=[3, 3], scale=2)
    net = BoundaryRefinementBlock(net, n_filters=21, kernel_size=[3, 3])

    fc_layer = tf.keras.layers.Conv2D(num_classes, kernel_size=[1, 1], padding='same', activation=None)
    net = fc_layer(net)
   
    return net, init_fn

def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)
