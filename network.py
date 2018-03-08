import tensorflow as tf
import numpy as np
sess = tf.Session()
data_size = 25
conv_size = 5
maxpool_size = 5
stride_size = 1
data_1d = np.random.normal(size=data_size)

x_input_1d = tf.placeholder(dtype = tf.float32,shape = [data_size])

def conv_layer_1d(input_1d,myfilter,stride):
    input_2d = tf.expand_dims(input_1d,0)
    input_3d = tf.expand_dims(input_2d,0)
    input_4d = tf.expand_dims(input_3d,3)

    convolution_output = tf.nn.conv2d(input_4d,filter=my_filter,strides=[1,1,stride,1],padding="VALID")
    conv_output_1d = tf.squeeze(convolution_output)
    return(conv_output_1d)

my_filter = tf.Variable(tf.random_normal(shape=[1,conv_size,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter,stride=stride_size)

