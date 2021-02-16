import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_caps, num_caps_prev, depth, depth_prev, routing_iterations=3, use_bias=True):
        super(CapsuleLayer, self).__init__()
        self.num_caps = num_caps
        self.num_caps_prev = num_caps_prev
        self.depth = depth
        self.depth_prev = depth_prev
        self.routing_iterations = routing_iterations
        self.use_bias = use_bias
        x = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        y = tf.zeros_initializer()
        if num_caps_prev is not None:
            self.weight_matrix = tf.Variable(x([1, num_caps_prev, num_caps, depth_prev, depth]), trainable=True)
            self.bias_matrix = tf.Variable(y([1, num_caps_prev, num_caps, depth]), trainable=True)
        else:
            self.weight_matrix = tf.Variable(x([1, 1, num_caps, depth_prev, depth]), trainable=True)
            self.bias_matrix = tf.Variable(y([1, 1, num_caps, depth]), trainable=True)

        # To access, W_ij = self.weight_matrices[0][i][j]

    def call(self, inp, **kwargs):  # inp is of shape (batch_size, num_caps_prev, depth_prev)
        batch_size, num_caps_prev = tf.shape(inp)[0], tf.shape(inp)[1]
        bij = tf.zeros([batch_size, num_caps_prev, self.num_caps])
        inp = tf.reshape(inp, [batch_size, num_caps_prev, 1, 1, self.depth_prev])

        # Here, we multiply W_ij * u_i
        weighted_input = tf.matmul(inp, self.weight_matrix)  # (batch_size, num_caps_prev, num_caps, 1, depth)
        weighted_input = tf.squeeze(weighted_input, axis=-2)  # (batch_size, num_caps_prev, num_caps, depth)
        if self.use_bias:
            weighted_input += self.bias_matrix
        weighted_input_gradients_stopped = tf.stop_gradient(weighted_input)

        vj = None
        for iteration in range(self.routing_iterations):
            cij = tf.keras.activations.softmax(bij, axis=-1)  # (batch_size, num_caps_prev, num_caps).
            cij = tf.expand_dims(cij, -1)  # (batch_size, num_caps_prev, num_caps, 1) "Coupling coefficients"

            if iteration == self.routing_iterations - 1:
                vj = cij * weighted_input  # (batch_size, num_caps_prev, num_caps, depth)
                vj = tf.reduce_sum(vj, axis=1)  # (batch_size, num_caps, depth)
                vj = self.squash(vj)  # (batch_size, num_caps, depth)
            else:
                vj = cij * weighted_input_gradients_stopped  # (batch_size, num_caps_prev, num_caps, depth)
                vj = tf.reduce_sum(vj, axis=1)  # (batch_size, num_caps, depth)
                vj = self.squash(vj)  # (batch_size, num_caps, depth)

                # updating logits
                bij += tf.reduce_sum(tf.expand_dims(vj, 1)
                                     * weighted_input_gradients_stopped,
                                     axis=-1)  # bij += (batch_size, num_caps_prev, num_caps)

        return vj

    def squash(self, inp):
        squared_magnitude_inp = tf.reduce_sum((inp ** 2), axis=-1)  # (batch_size, num_caps)
        squared_magnitude_inp = tf.expand_dims(squared_magnitude_inp, -1)  # (batch_size, num_caps, 1)
        scale = (squared_magnitude_inp / (1 + squared_magnitude_inp))  # (batch_size, num_caps, 1)
        inp = scale * inp / tf.math.sqrt(squared_magnitude_inp)  # (batch_size, num_caps, depth)
        return inp


class ConvolutionalCapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters_prev, depth_prev, filters, depth):
        super(ConvolutionalCapsuleLayer, self).__init__()

        self.kernel_size = kernel_size
        self.filters_prev = filters_prev
        self.depth_prev = depth_prev
        self.filters = filters
        self.depth = depth

        x = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        y = tf.zeros_initializer()
        self.weight_matrix = tf.Variable(x([1, 1, kernel_size * filters_prev,
                                            filters, depth_prev, depth]), trainable=True)
        self.bias_mat = tf.Variable(y([1, 1, kernel_size * filters_prev, filters, depth]), trainable=True)

    def call(self, input, **kwargs):
        # inp shape: (None, seq_len, filters_prev, depth_prev)
        inp_shape = tf.shape(input)
        conv_caps = tf.TensorArray(dtype=tf.float32, size=inp_shape[1] - self.kernel_size + 1, dynamic_size=False)
        for i in tf.range(inp_shape[1] - self.kernel_size + 1):
            conv_caps = conv_caps.write(i, tf.reshape(input[:, i:i + self.kernel_size, :, :],
                                                      [inp_shape[0], self.kernel_size * inp_shape[2], inp_shape[3]]))
        conv_caps = conv_caps.stack()
        # conv_caps = tf.transpose(conv_caps, perm=[1, 0, 2, 3])
        # conv_caps_shape: (None, seq_len - kernel_size + 1, kernel_size * filters_prev, depth_prev)

        c_s = tf.shape(conv_caps)
        conv_caps = tf.reshape(conv_caps, [c_s[0], c_s[1], c_s[2], 1, 1, c_s[3]])
        conv_caps = tf.matmul(conv_caps, self.weight_matrix)
        conv_caps = tf.squeeze(conv_caps, axis=-2)
        conv_caps += self.bias_mat
        # conv_caps shape: (None, seq_len - kernel_size + 1, kernel_size * filters_prev, filters, depth)

        b_ij = tf.zeros([c_s[0], c_s[1], c_s[2], self.filters])
        # b_ij shape:      (None, seq_len - kernel_size + 1, kernel_size * filters_prev, filters)

        r = 3  # routing iterations
        vj = None
        for iteration in range(r):
            cij = tf.keras.activations.softmax(b_ij, axis=-1)
            cij = tf.expand_dims(cij, -1)

            if iteration == r - 1:
                vj = cij * conv_caps
                vj = tf.reduce_sum(vj, axis=2)
                vj = self.squash(vj)  # (None, seq_len - kernel_size + 1, filters, depth)
            else:
                vj = cij * conv_caps
                vj = tf.reduce_sum(vj, axis=2)
                vj = self.squash(vj)  # (None, seq_len - kernel_size + 1, filters, depth)

                # updating logits
                b_ij += tf.reduce_sum(tf.expand_dims(vj, 2)
                                      * conv_caps, axis=-1)  # bij += (batch_size, num_caps_prev, num_caps)

        return tf.transpose(vj, perm=[1, 0, 2, 3])


    def squash(self, inp):
        squared_magnitude_inp = tf.reduce_sum((inp ** 2), axis=-1)  # (batch_size, num_caps)
        squared_magnitude_inp = tf.expand_dims(squared_magnitude_inp, -1)  # (batch_size, num_caps, 1)
        scale = (squared_magnitude_inp / (1 + squared_magnitude_inp))  # (batch_size, num_caps, 1)
        inp = scale * inp / tf.math.sqrt(squared_magnitude_inp)  # (batch_size, num_caps, depth)
        return inp


class CapsNet(tf.keras.Model):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=256, kernel_size=[9,9], activation="relu",
                                           input_shape=(28, 28, 1), data_format="channels_last")
        self.primary_caps = tf.keras.layers.Conv2D(filters=256, kernel_size=[9, 9], activation="relu",
                                                   strides=(2, 2), data_format="channels_last")
        self.digit_caps = CapsuleLayer(10, 6 * 6 * 32, 16, 8)

    def call(self, inp):
        # inp = tf.expand_dims(inp, axis=-1)  # (batch_size, 28, 28, 1).
        conv1 = self.conv(inp)  # (batch_size, 20, 20, 256)
        # conv1 = tf.expand_dims(conv1, axis=-1)
        primary_caps = self.primary_caps(conv1)  # (batch_size, 6, 6, 256)
        primary_caps = tf.reshape(primary_caps, [-1, 1152, 8])
        digit_caps = self.digit_caps(primary_caps)
        return digit_caps

    def squash(self, inp):
        squared_magnitude_inp = tf.reduce_sum((inp ** 2), axis=-1)  # (batch_size, num_caps)
        squared_magnitude_inp = tf.expand_dims(squared_magnitude_inp, -1)  # (batch_size, num_caps, 1)
        scale = (squared_magnitude_inp / (1 + squared_magnitude_inp))  # (batch_size, num_caps, 1)
        inp = scale * inp / tf.math.sqrt(squared_magnitude_inp)  # (batch_size, num_caps, depth)
        return inp


# tested working
def squash(inp):
    squared_magnitude_inp = tf.reduce_sum((inp ** 2), axis=-1)  # (batch_size, num_caps)
    squared_magnitude_inp = tf.expand_dims(squared_magnitude_inp, -1)  # (batch_size, num_caps, 1)
    scale = (squared_magnitude_inp / (1 + squared_magnitude_inp))  # (batch_size, num_caps, 1)
    inp = scale * inp / tf.math.sqrt(squared_magnitude_inp)  # (batch_size, num_caps, depth)
    return inp

