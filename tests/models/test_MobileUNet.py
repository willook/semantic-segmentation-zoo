import unittest
import numpy as np
import tensorflow as tf
from MobileUNet import MobileUNet, ConvBlock, DepthwiseSeparableConvBlock, ConvTransposeBlock, DownSamplingBlock, UpSamplingBlock

class TestStringMethods(unittest.TestCase):

    def get_random_data(self, shape):
        return tf.random.uniform([*shape])

    def network(self, input_shape, num_classes):
        output_shape = input_shape[:-1] + (num_classes,)
        inputs = self.get_random_data(input_shape)
        net = MobileUNet(num_classes, "MobileUNet")

        output = net(inputs)
        return output, output_shape

    def network_skip(self, input_shape, num_classes):
        output_shape = input_shape[:-1] + (num_classes,)
        inputs = self.get_random_data(input_shape)
        net = MobileUNet(num_classes)

        output = net(inputs)
        return output, output_shape

    def test_ConvBlock(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = input_shape[:-1] + (num_classes,)

        inputs = self.get_random_data(input_shape)
        net = ConvBlock(num_classes)

        output = net(inputs)
        self.assertEqual(output_shape, output.shape)

    def test_DepthwiseSeparableConvBlock(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = input_shape[:-1] + (num_classes,)

        inputs = self.get_random_data(input_shape)
        net = DepthwiseSeparableConvBlock(num_classes)

        output = net(inputs)
        self.assertEqual(output_shape, output.shape)

    def test_ConvTransposeBlock(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = (input_shape[0], input_shape[1]*2, input_shape[2]*2) + (num_classes,)

        inputs = self.get_random_data(input_shape)
        net = ConvTransposeBlock(num_classes)

        output = net(inputs)
        self.assertEqual(output_shape, output.shape)    

    def test_DownSamplingBlock(self):
        num_classes = 16
        n = 3
        input_shape = (1,32,32,3)
        output_shape = (input_shape[0], int(input_shape[1]/2), int(input_shape[2]/2)) + (num_classes,)

        inputs = self.get_random_data(input_shape)
        net = DownSamplingBlock(num_classes, n)

        output = net(inputs)
        self.assertEqual(output_shape, output.shape)

    def test_UpSamplingBlock_skip(self):
        num_classes = 16
        n = 3
        input_shape = (1,32,32,3)
        skip_shape = (1,64,64,num_classes)
        output_shape = skip_shape[:-1] + (num_classes,)

        inputs = self.get_random_data(input_shape)
        skips = self.get_random_data(skip_shape)
        net = UpSamplingBlock(num_classes, n, is_skip=True, out_depthwise_ch=num_classes)

        output = net(inputs, skips)
        self.assertEqual(output_shape, output.shape) 

    def test_UpSamplingBlock(self):
        num_classes = 16
        n = 3
        input_shape = (1,32,32,3)
        output_shape = (input_shape[0], input_shape[1]*2, input_shape[2]*2) + (num_classes,)

        inputs = self.get_random_data(input_shape)
        net = UpSamplingBlock(num_classes, n, is_skip=False, out_depthwise_ch=num_classes)

        output = net(inputs, None)
        self.assertEqual(output_shape, output.shape) 
    
    def test_UpSamplingBlock_padding(self):
        num_classes = 4
        n = 3

        input_shape1 = (1,32,32,num_classes)
        skip_shape1 = (1,64,65,num_classes)
        output_shape1 = skip_shape1

        input_shape2 = (1,31,31,num_classes)
        skip_shape2 = (1,63,62,num_classes)
        output_shape2 = skip_shape2

        input_shape3 = (1,1,1,num_classes)
        skip_shape3 = (1,2,2,num_classes)
        output_shape3 = skip_shape3

        net1 = UpSamplingBlock(num_classes, n, is_skip=True, out_depthwise_ch=num_classes)
        net2 = UpSamplingBlock(num_classes, n, is_skip=True, out_depthwise_ch=num_classes)
        net3 = UpSamplingBlock(num_classes, n, is_skip=True, out_depthwise_ch=num_classes)

        inputs1 = self.get_random_data(input_shape1)
        inputs2 = self.get_random_data(input_shape2)
        inputs3 = self.get_random_data(input_shape3)

        skips1 = self.get_random_data(skip_shape1)
        skips2 = self.get_random_data(skip_shape2)
        skips3 = self.get_random_data(skip_shape3)

        output1 = net1(inputs1, skips1)
        output2 = net2(inputs2, skips2)
        output3 = net3(inputs3, skips3)

        self.assertEqual(output_shape1, output1.shape)
        self.assertEqual(output_shape2, output2.shape)
        self.assertEqual(output_shape3, output3.shape)

    def test_output_type_skip(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = input_shape[:-1] + (num_classes,)
        inputs = self.get_random_data(input_shape)
        net = MobileUNet(num_classes)

        output = net(inputs)

        self.assertIsInstance(output, tf.Tensor)
        self.assertIs(output.dtype, tf.float32)
        self.assertEqual(output_shape, output.shape)

    def test_nomal_input_skip(self):
        num_classes = 32
        input_shape = (1,512,512,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_multi_batch_skip(self):
        num_classes = 32
        input_shape = (3,64,64,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_variable_input_skip(self):
        num_classes = 32
        input_shape = (1,71,61,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,117,41,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,37,129,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_small_input_skip(self):
        num_classes = 32
        input_shape = (1,15,15,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,15,512,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,512,15,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,1,1,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_variable_num_classes_skip(self):
        num_classes = 16
        input_shape = (1,512,512,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 7
        input_shape = (1,512,512,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 2
        input_shape = (1,512,512,3)
        output, output_shape = self.network_skip(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_output_type(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = input_shape[:-1] + (num_classes,)
        inputs = self.get_random_data(input_shape)
        net = MobileUNet(num_classes, "MobileUNet")

        output = net(inputs)

        self.assertIsInstance(output, tf.Tensor)
        self.assertIs(output.dtype, tf.float32)
        self.assertEqual(output_shape, output.shape)

    def test_nomal_input(self):
        num_classes = 32
        input_shape = (1,512,512,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_multi_batch(self):
        num_classes = 32
        input_shape = (3,64,64,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_variable_input(self):
        num_classes = 32
        input_shape = (1,71,61,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,117,41,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,37,129,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_small_input(self):
        num_classes = 32
        input_shape = (1,15,15,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,15,512,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,512,15,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 32
        input_shape = (1,1,1,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

    def test_variable_num_classes(self):
        num_classes = 16
        input_shape = (1,512,512,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 7
        input_shape = (1,512,512,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

        num_classes = 2
        input_shape = (1,512,512,3)
        output, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, output.shape)

if __name__ == '__main__':
    unittest.main()