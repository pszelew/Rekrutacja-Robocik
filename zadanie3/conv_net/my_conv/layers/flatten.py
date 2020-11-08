from .layer import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self, input_dims: tuple, output_dims: tuple):
        super().__init__("flatten", input_dims, output_dims)
    def forward(self, input: np.array):
        """Flattening operation

        Parameters
        ----------
        arr: np.array
            Input array of flattening operation 
        Returns
        -------
        np.array
            Flattened array
        """
        out_arr: np.array
        batch_size: int
        batch_size = input.shape[0]
        out_arr = np.reshape(input, (batch_size, -1))
        self.activation_val = self.output = out_arr
        return out_arr
    def back(self, prop: np.array, prev_output: np.array, next_weights: np.array, lr: np.float64):
        loss_derivative: np.array
        loss_derivative = np.zeros(self.output_dims)
        # out_derivative: np.array
        # activation_derivative: np.array
        # print(next_weights.shape)
        for data in range(prop.shape[0]):
            for i in range(len(loss_derivative)):
                # For all neurons of current layer
                for j in range(len(prop)):
                    # Loss propagating from next layer
                    loss_derivative[data][i] += next_weights[j][i] * prop[data][j]
                    # Temp_der times weight connecting it to our neuron 

        # out_derivative = self.activation_val
        # b) Calculate delta(lay_out)/delta(activation)

        # activation_derivative = self.activation_derivative(weights, prev_output)
        # c) Calculate delta(activation)/delta(weights)
        #print(prev_output.shape)
        prop = np.reshape(loss_derivative, (prev_output.shape[0], prev_output.shape[1], prev_output.shape[2], -1))
        #print(prop.shape)
        return prop