from .layer import Layer
import numpy as np
from numpy import savetxt

class Flatten(Layer):
    """
    A  class used to represent flatten layer
    
    Methods
    -------
    forward(arr: np.array) -> np.array
        Returns result of forward propagation
    back(prop: np.array, prev_output: np.array) -> np.array
        Back propagate error derivative
    """
    def __init__(self, input_dims: tuple, output_dims: tuple):
        """
        Parameters
        ----------
        input_dims: tuple
            Shape of input tensor
        output_dims: tuple
            Shape of output tensor
        """
        super().__init__("flatten", input_dims, output_dims)
    def forward(self, input: np.array):
        """Flatten network forward pass

        Parameters
        ----------
        arr: np.array
            Input array of flattening operation for eg. 4D np.array
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
    def back(self, prop: np.array, prev_output: np.array, next_weights: np.array):
        """Flatten layer back propagation

        Parameters
        ----------
        prop: np.array
            Loss error derivative in output of the NEXT layer. Shape -- next layer output_dims

        Returns
        -------
        np.array
            Result of flatten backpropagation. Shape -- input_dims
            Thats an error on PREVIOUS layer neurons!
        """   
        loss_derivative: np.array
        loss_derivative = np.zeros(self.output_dims)
        # out_derivative: np.array
        # activation_derivative: np.array
        
        for data in range(prop.shape[0]):
            for i in range(len(loss_derivative)):
                # For all neurons of current layer
                for j in range(len(prop)):
                    # For all neurons from next layer
                    loss_derivative[data][i] += next_weights[j][i] * prop[data][j]
                    # Temp_der times weight connecting next to our neuron 

        # out_derivative = self.activation_val
        # b) Calculate delta(lay_out)/delta(activation)

        # activation_derivative = self.activation_derivative(weights, prev_output)
        # c) Calculate delta(activation)/delta(weights)
        # print(prev_output.shape)
        prop = np.reshape(loss_derivative, (prev_output.shape[0], prev_output.shape[1], prev_output.shape[2], -1))
        # print(loss_derivative.max())
        # print("Flat prop -- saving to file")
        # savetxt('prop_flat.csv', prop[0, :, :, 55], delimiter=',')
        return prop