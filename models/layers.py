from typing import Tuple, Optional
from numpy.lib.stride_tricks import as_strided
import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:

    def forward(self, inputs, training):
        self.output = inputs


class FlattenLayer:
    def __init__(self):
        self._shape = ()
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self._shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self._shape)



class ConvolutionalLayer:
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding=0, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(filters, in_channels, kernel_size, kernel_size) * np.sqrt(
            2. / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros(filters)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, input_tensor, training):
        self.inputs = np.pad(input_tensor, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        batch_size, in_channels, in_height, in_width = self.inputs.shape
        out_height = int((in_height - self.kernel_size) / self.stride + 1)
        out_width = int((in_width - self.kernel_size) / self.stride + 1)

        self.output = np.zeros((batch_size, self.filters, out_height, out_width))

        input_strided = as_strided(self.inputs,
                                   shape=(batch_size, in_channels, out_height, out_width, self.kernel_size, self.kernel_size),
                                   strides=(*self.inputs.strides[:2], self.inputs.strides[2]*self.stride, self.inputs.strides[3]*self.stride, *self.inputs.strides[2:]))

        for k in range(self.filters):
            self.output[:, k, :, :] = np.tensordot(input_strided, self.weights[k, :, :, :], axes=([1, 4, 5], [0, 1, 2])) + self.biases[k]

        self.output = np.maximum(0, self.output)
        return self.output

    def backward(self, output_error):
        output_error = output_error * (self.output > 0)  # Apply derivative of ReLU to error
        batch_size, in_channels, in_height, in_width = self.inputs.shape
        _, _, out_height, out_width = output_error.shape

        dweights = np.zeros_like(self.weights)
        dbiases = np.zeros_like(self.biases)
        dinputs = np.zeros_like(self.inputs)

        input_strided = as_strided(self.inputs,
                                   shape=(
                                   batch_size, in_channels, out_height, out_width, self.kernel_size, self.kernel_size),
                                   strides=(
                                   self.inputs.strides[0], self.inputs.strides[1], self.inputs.strides[2] * self.stride,
                                   self.inputs.strides[3] * self.stride, self.inputs.strides[2],
                                   self.inputs.strides[3]))

        for k in range(self.filters):
            expanded_output_error = output_error[:, k, :, :].reshape(batch_size, 1, out_height, out_width, 1, 1)
            dweights[k] = np.sum(input_strided * expanded_output_error, axis=(0, 2, 3))
            dbiases[k] = np.sum(output_error[:, k, :, :], axis=(0, 1, 2))


        flipped_weights = self.weights[:, :, ::-1, ::-1]
        padded_error = np.pad(output_error,
                              ((0, 0), (0, 0),
                               (self.kernel_size - 1, self.kernel_size - 1),
                               (self.kernel_size - 1, self.kernel_size - 1)),
                              mode='constant')


        strided_error = as_strided(padded_error,
                                   shape=(
                                   batch_size, self.filters, in_height, in_width, self.kernel_size, self.kernel_size),
                                   strides=(padded_error.strides[0], padded_error.strides[1],
                                            padded_error.strides[2] * self.stride,
                                            padded_error.strides[3] * self.stride,
                                            padded_error.strides[2], padded_error.strides[3]))


        dinputs = np.tensordot(strided_error, flipped_weights, axes=[[1, 4, 5], [0, 2, 3]])

        if self.padding != 0:
            dinputs = dinputs[:, :, self.padding:-self.padding, self.padding:-self.padding]

        self.dweights = dweights
        self.dbiases = dbiases
        self.dinputs = dinputs

        return dinputs


class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.inputs = None
        self.output = None

    def forward(self, x, training):
        self.inputs = x
        batch_size, channels, height, width = x.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        strided_input = as_strided(x,
                                   shape=(batch_size, channels, out_height, out_width, self.pool_size, self.pool_size),
                                   strides=(
                                   x.strides[0], x.strides[1], x.strides[2] * self.stride, x.strides[3] * self.stride,
                                   x.strides[2], x.strides[3]))

        self.output = np.max(strided_input, axis=(4, 5))

        return self.output

    def backward(self, output_error):
        batch_size, channels, height, width = self.inputs.shape
        out_height, out_width = output_error.shape[2], output_error.shape[3]

        strided_input = as_strided(self.inputs,
                                   shape=(batch_size, channels, out_height, out_width, self.pool_size, self.pool_size),
                                   strides=(self.inputs.strides[0], self.inputs.strides[1], self.inputs.strides[2] * self.stride, self.inputs.strides[3] * self.stride, self.inputs.strides[2], self.inputs.strides[3]))

        input_error = np.zeros_like(self.inputs)

        for i in range(out_height):
            for j in range(out_width):
                max_indices = np.argmax(strided_input[:, :, i, j].reshape(batch_size, channels, -1), axis=2)

                max_coords = np.unravel_index(max_indices, (self.pool_size, self.pool_size))

                for b in range(batch_size):
                    for c in range(channels):
                        h, w = max_coords[0][b, c], max_coords[1][b, c]
                        h_start, w_start = i * self.stride, j * self.stride
                        input_error[b, c, h_start + h, w_start + w] += output_error[b, c, i, j]

        self.dinputs = input_error
        return input_error
