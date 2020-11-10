from .layer import Layer
import numpy as np
from numpy import savetxt


class MaxPooling(Layer):
    """
    A  class used to represent MaxPooling layer
    
    Methods
    -------
    forward(arr: np.array) -> np.array
        Returns result of forward propagation
    back(prop: np.array, prev_output: np.array) -> np.array
        Back propagate error derivative
    """
    def __init__(self, input_dims: tuple, output_dims: tuple, n: int = 2):
        """
        Parameters
        ----------
        input_dims: tuple
            Shape of input tensor
        output_dims: tuple
            Shape of output tensor
        n: int
            Size of window (default: 2)
        """
        super().__init__("max_pooling", input_dims, output_dims)
        self.n = n
    def forward(self, arr: np.array) -> np.array:
        """Max-pooling operation

        Parameters
        ----------
        arr: np.array
            Input array of max-pooling operation

        Returns
        -------
        np.array
            Result of max-pooling operation
        """
        # print(arr.shape)
        temp_shape = arr.shape
        out_dims = (temp_shape[0], temp_shape[1]//self.n, temp_shape[2]//self.n, temp_shape[3])
        # Dimensions of output
        out_arr: np.array
        out_arr = np.zeros(out_dims)
        # Our output array
        for i, image in enumerate(arr):
            # Here we have got infromation from every image from our batch.
            # Let's do max-polling on this image data
            # print(image.shape)
            for j in range(image.shape[2]):
                # For every channel
                slice = image[..., j]
                # Now we have slice we are only interested in
                # print(slice.shape)
                # Just for debug purpose
                for k in range(out_dims[1]):
                    for l in range(out_dims[2]):
                        out_arr[i][k][l][j] = np.max(slice[k*self.n:(k+1)*self.n, l*self.n:(l+1)*self.n])
        self.activation_val = self.output = out_arr
        return out_arr
    def back(self, prop: np.array, prev_output: np.array) -> np.array:
        """Max-pooling back propagation

        Parameters
        ----------
        prop: np.array
            Loss error derivative in layer output
        prev_output: np.array
            Output of previous layer
        Returns
        -------
        np.array
            Result of max-pooling backpropagation. shape -- (input_dims).
            Thats an error on PREVIOUS layer neurons!
        """   
        #  print("Start max pooling")
        inverse_max_pooling:np.array
        # print(self.input_dims)
        inverse_max_pooling = np.zeros(self.input_dims)
        # print(prop.shape)
        # print(next_weights.shape)
        for batch in range(inverse_max_pooling.shape[0]): 
            for channel in range(inverse_max_pooling.shape[3]):
                for i in range(self.output_dims[1]): #neuron num
                    for j in range(self.output_dims[2]): #neuron num
                        for k in range(self.n): #slide x pooling
                            for l in range(self.n): #slide y pooling
                                if self.output[batch][i][j][channel] == self.input[batch][k+i*2][l+j*2][channel]:
                                    # print(inverse_max_pooling.shape)
                                    # print(prop.shape)
                                    # print(self.output_dims)
                                    inverse_max_pooling[batch][k+i*2][l+j*2][channel] += prop[batch][i][j][channel]
        return inverse_max_pooling