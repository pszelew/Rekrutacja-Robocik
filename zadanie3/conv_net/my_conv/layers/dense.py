from conv import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, activation_fcn: function, input_dims: tuple, output_dims: tuple):
        super.__init__("dense", input_dims, output_dims)
        self.activation_fcn = activation_fcn
    def forward(self, input: np.array, weights: np.array, bias: np.array):
        """Dense layer of neural network
        Parameters
        ----------
        activation_fcn: function
            Activation function of layer
        input: np.array
            Input array of max-pooling operation
        weights: np.array
            Array of weights
        bias: np.array
            Added bias
        Returns
        -------
        np.array
            Output of the dense layer
        """
        out_dims: tuple
        out_dims = (input.shape[0], weights.shape[1])
        out_arr: np.array
        out_arr = np.zeros(out_dims)
        
        for i, data in enumerate(input):
            out_arr[i] = self.activation_fcn(np.matmul(weights, data) + bias)
        self.activation_val = out_arr
        return out_arr