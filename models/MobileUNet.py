import os,time,cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D, Conv2DTranspose, MaxPool2D

class ConvBlock(tf.keras.layers.Layer):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	def __init__(self, n_filters, kernel_size=[3, 3]):
		super(ConvBlock, self).__init__()
		self.conv2d = Conv2D(n_filters, kernel_size=[1, 1], padding='same', activation=None)
		self.batchnorm = BatchNormalization(fused=True)
		self.relu = ReLU()

	def call(self, x):
		x = self.conv2d(x)
		x = self.batchnorm(x)
		x = self.relu(x)
		return x

class DepthwiseSeparableConvBlock(tf.keras.layers.Layer):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	def __init__(self, n_filters, kernel_size=[3, 3]):
		super(DepthwiseSeparableConvBlock, self).__init__()
		self.separableconv2d = SeparableConv2D(n_filters, depth_multiplier=1, kernel_size=[3, 3], padding='same', activation=None)
		self.batchnorm = BatchNormalization(fused=True)
		self.relu = ReLU()
		self.conv2d = Conv2D(n_filters, kernel_size=[1, 1], padding='same', activation=None)

	def call(self, x):
		x = self.separableconv2d(x)
		x = self.batchnorm(x)
		x = self.relu(x)
		x = self.conv2d(x)
		x = self.batchnorm(x)
		x = self.relu(x)
		return x

class ConvTransposeBlock(tf.keras.layers.Layer):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	def __init__(self, n_filters, kernel_size=[3, 3]):
		super(ConvTransposeBlock, self).__init__()
		self.conv2dtranspose = Conv2DTranspose(n_filters, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=None)
		self.batchnorm = BatchNormalization()
		self.relu = ReLU()

	def call(self, x):
		x = self.conv2dtranspose(x)
		x = self.batchnorm(x)
		x = self.relu(x)
		return x

class ResidualBlock(tf.keras.layers.Layer):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	def __init__(self, n_filters, n, isTranspose):
		super(ResidualBlock, self).__init__()
		self.features = tf.keras.Sequential()
		if isTranspose:
			self.features.add(ConvTransposeBlock(n_filters))
			for _ in range(n):
				self.features.add(DepthwiseSeparableConvBlock(n_filters))
		else:
			for _ in range(n):
				self.features.add(DepthwiseSeparableConvBlock(n_filters))
			self.features.add(MaxPool2D((2, 2), strides=2))

	def call(self, x):
		return self.features(x)


class build_mobile_unet(tf.keras.Model):
	def __init__(self, preset_model, num_classes):
		super(build_mobile_unet, self).__init__()
		self.conv_block = ConvBlock(64)
		self.downsampling_block1 = ResidualBlock(64, 1, False)
		self.downsampling_block2 = ResidualBlock(128, 2, False)
		self.downsampling_block3 = ResidualBlock(256, 3, False)
		self.downsampling_block4 = ResidualBlock(512, 3, False)
		self.downsampling_block5 = ResidualBlock(512, 3, False)
		self.upsampling_block1 = ResidualBlock(512, 3, True)
		self.upsampling_block2 = ResidualBlock(512, 2, True)
		self.upsampling_block3 = ResidualBlock(256, 2, True)
		self.upsampling_block4 = ResidualBlock(128, 1, True)
		self.upsampling_block5 = ResidualBlock(64, 2, True)
		
		self.depthwise_separable_conv_block_256 = DepthwiseSeparableConvBlock(256)
		self.depthwise_separable_conv_block_128 = DepthwiseSeparableConvBlock(128)
		self.depthwise_separable_conv_block_64 = DepthwiseSeparableConvBlock(64)
		self.fc = Conv2D(num_classes, kernel_size=[1, 1], padding='same', activation=None)
		self.has_skip = False

		if preset_model == "MobileUNet":
			self.has_skip = False
		elif preset_model == "MobileUNet-Skip":
			self.has_skip = True
		else:
			raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

	def call(self, inputs):
		#####################
		# Downsampling path #
		#####################
		net = self.conv_block(inputs)
		net = self.downsampling_block1(net)
		skip_1 = net

		net = self.downsampling_block2(net)
		skip_2 = net
		
		net = self.downsampling_block3(net)
		skip_3 = net
		
		net = self.downsampling_block4(net)
		skip_4 = net
		
		net = self.downsampling_block5(net)
		
		#####################
		# Upsampling path #
		#####################
		net = self.upsampling_block1(net)
		if self.has_skip:
			net = tf.add(net, skip_4)

		net = self.upsampling_block2(net)
		net = self.depthwise_separable_conv_block_256(net)
		if self.has_skip:
			net = tf.add(net, skip_3)

		net = self.upsampling_block3(net)
		net = self.depthwise_separable_conv_block_128(net)
		if self.has_skip:
			net = tf.add(net, skip_2)

		net = self.upsampling_block4(net)
		net = self.depthwise_separable_conv_block_64(net)
		if self.has_skip:
			net = tf.add(net, skip_1)

		net = self.upsampling_block5(net)

		#####################
		#      Softmax      #
		#####################
		net = self.fc(net)
		return net