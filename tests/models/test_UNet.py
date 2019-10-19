import unittest
import numpy as np
import tensorflow as tf
from models.UNet import build_unet, ConvBlock, conv_transpose_block

class TestStringMethods(unittest.TestCase):

    def get_random_data(self, shape):
        return np.random.rand(*shape)

    def network(self, input_shape, num_classes):
        output_shape = input_shape[:-1] + (num_classes,)
        input_data = self.get_random_data(input_shape)
        net_input = tf.placeholder(tf.float32, shape=input_shape)
        net = build_unet(net_input, "UNet", num_classes)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        net = sess.run(net, feed_dict = {net_input: input_data})
        return net, output_shape
    
    def test_ConvBlock(self):
        num_classes = 16
        input_shape = (1,32,32,4)
        output_shape = input_shape[:-1] + (num_classes,)
        
        net_input = tf.placeholder(tf.float32, shape=input_shape)
        net = ConvBlock(net_input, num_classes)
        
        input_data = self.get_random_data(input_shape)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(net, feed_dict = {net_input: input_data})
        self.assertEqual(output_shape, output.shape)
    
    def test_conv_transpose_block(self):
        num_classes = 4
        input_shape = (1,32,32,num_classes)
        output_shape = (1,64,64,num_classes)
        
        net_input = tf.placeholder(tf.float32, shape=input_shape)
        net = conv_transpose_block(net_input, num_classes)
        
        input_data = self.get_random_data(input_shape)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(net, feed_dict = {net_input: input_data})
        self.assertEqual(output_shape, output.shape)

    def test_conv_transpose_padding(self):
        num_classes = 4
        input_shape1 = (1,32,32,num_classes)
        output_shape1 = (1,63,64,num_classes)

        input_shape2 = (1,31,31,num_classes)
        output_shape2 = (1,61,62,num_classes)

        input_shape3 = (1,1,1,num_classes)
        output_shape3 = (1,1,1,num_classes)

        input_shape4 = (1,1,1,num_classes)
        output_shape4 = (1,2,2,num_classes)
        
        net_input1 = tf.placeholder(tf.float32, shape=input_shape1)
        net_input2 = tf.placeholder(tf.float32, shape=input_shape2)
        net_input3 = tf.placeholder(tf.float32, shape=input_shape3)
        net_input4 = tf.placeholder(tf.float32, shape=input_shape4)
        
        net1 = conv_transpose_block(net_input1, num_classes,
                                    output_shape=output_shape1)
        net2 = conv_transpose_block(net_input2, num_classes,
                                    output_shape=output_shape2)
        net3 = conv_transpose_block(net_input3, num_classes,
                                    output_shape=output_shape3)
        net4 = conv_transpose_block(net_input4, num_classes,
                                    output_shape=output_shape4)
        
        input_data1 = self.get_random_data(input_shape1)
        input_data2 = self.get_random_data(input_shape2)
        input_data3 = self.get_random_data(input_shape3)
        input_data4 = self.get_random_data(input_shape4)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output_list = sess.run([net1, net2, net3, net4],
            feed_dict = {net_input1: input_data1, net_input2: input_data2,
                         net_input3: input_data3, net_input4: input_data4})
       
        self.assertEqual(output_shape1, output_list[0].shape)
        self.assertEqual(output_shape2, output_list[1].shape)
        self.assertEqual(output_shape3, output_list[2].shape)
        self.assertEqual(output_shape4, output_list[3].shape)

    def test_conv_transpose_strides(self):
        num_classes = 4
        input_shape = (1,32,31,num_classes)
        output_shape = (1,95,121,num_classes)
        strides = [3, 4]
        
        net_input = tf.placeholder(tf.float32, shape=input_shape)
        net = conv_transpose_block(net_input, num_classes, strides=strides,
            output_shape=output_shape)
        
        input_data = self.get_random_data(input_shape)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(net, feed_dict = {net_input: input_data})
        self.assertEqual(output_shape, output.shape)
    
    def test_output_type(self):
        num_classes = 16
        input_shape = (1,32,32,3)
        output_shape = input_shape[:-1] + (num_classes,)
        input_data = self.get_random_data(input_shape)
        net_input = tf.placeholder(tf.float32, shape=input_shape)
        net = build_unet(net_input, "UNet", num_classes)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(net, feed_dict = {net_input: input_data})
        
        self.assertIsInstance(net, tf.Tensor)
        self.assertIsInstance(output, np.ndarray)
        self.assertIs(net.dtype, tf.float32)
        self.assertIs(output.dtype, np.dtype('float32'))
        
    def test_nomal_input(self):
        num_classes = 32
        input_shape = (1,512,512,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

    def test_multi_batch(self):
        num_classes = 32
        input_shape = (3,64,64,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)
        
    def test_variable_input(self):
        num_classes = 32
        input_shape = (1,71,61,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 32
        input_shape = (1,117,41,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 32
        input_shape = (1,37,129,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)
        
    def test_small_input(self):
        num_classes = 32
        input_shape = (1,15,15,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 32
        input_shape = (1,15,512,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 32
        input_shape = (1,512,15,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 32
        input_shape = (1,1,1,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)
        
    def test_variable_num_classes(self):
        num_classes = 16
        input_shape = (1,512,512,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 7
        input_shape = (1,512,512,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

        num_classes = 2
        input_shape = (1,512,512,3)
        net, output_shape = self.network(input_shape, num_classes)
        self.assertEqual(output_shape, net.shape)

    
if __name__ == '__main__':
    unittest.main()
