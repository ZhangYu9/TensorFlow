
import tensorflow as tf
import numpy as np
sess = tf.Session()
data_size = 25
data_1d = np.random.normal(size=data_size)

x_input_1d = tf.placeholder(dtype = tf.float32,shape=[data_size])


def conv_layer_1d(input_1d, my_filter):

    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input_4d,filter=my_filter,strides=[1,1,1,1],padding="VALID")
    conc_output_1d = tf.squeeze(convolution_output)
    return conc_output_1d
my_filter = tf.Variable(tf.random_normal(shape = [1, 5, 1, 1]))
my_convolution_output = conv_layer_1d(x_input_1d,my_filter)


def activation(input_1d):
    return tf.nn.relu(input_1d)
my_activation_output = activation(my_convolution_output)

#--------Max Pool--------
def max_pool(input_1d, width):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, width=1, height=num_input, channels=1]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 1, 2, 1]
    # We will also need to specify the width of the max-window ('width')
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    # Get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return(pool_output_1d)

my_maxpool_output = max_pool(my_activation_output, width=5)


def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer)]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    input_layer_2d = tf.expand_dims(input_layer, 0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    full_output_1d = tf.squeeze(full_output)
    return full_output_1d
my_pull_output = fully_connected(my_maxpool_output, 5)

init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d:data_1d}
print('Input = array of length 25')
print('Convolution w/filter,length=5,stride_size = 1,results in array of length 21:')
print(sess.run(my_convolution_output,feed_dict=feed_dict))
print('\nInput = the above array of length 21')
print('ReLU element size return the array of length 21:')
print(sess.run(my_activation_output,feed_dict=feed_dict))
print('\nInput = the above array of length 21')
print('MaxPool, window length =5 stride_size=1,results in the array of length 17:')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
print(sess.run(my_pull_output, feed_dict=feed_dict))
