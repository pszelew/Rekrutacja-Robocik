from __future__ import annotations
from conv import Layer
import numpy as np


class Conv(Layer):
    def __init__(self, input_dims: tuple, output_dims: tuple):
        super.__init__("conv", input_dims, output_dims)
        
    def forward(self, input: np.array, weights: np.array, bias: np.array):
        """Conv net operation
        Parameters
        ----------
        input: np.array
            Input array of max-pooling operation
        weights: np.array
            Array of weights
        bias: np.array
            Added bias
        Returns
        -------
        np.array
            Output of the conv layer
        """
        n = weights.shape[0]
        # Window size
        print(input.shape)
        temp_shape = input.shape
        out_dims = (temp_shape[0], temp_shape[1]-weights.shape[0]+1, temp_shape[2]-weights.shape[1]+1, weights.shape[3])
        # Dimensions of output. They are defined by weight array
        out_arr: np.array
        out_arr = np.zeros(out_dims)
        # Our output array
        for i, image in enumerate(input):
            # Here we have got infromation from every image from our batch.
            # Let's do our conv magic on this image data
            # print(image.shape)
            for j in range(weights.shape[3]):
                # For every channel
                filt = weights[..., j]
                # Now we have slice we are only interested in. Filt is our filter
                # print(slice.shape)
                # Just for debug purpose

                for k in range(out_dims[1]):
                    for l in range(out_dims[2]):
                        # Now let's calculate result for our filter
                        res: np.float64
                        res = np.sum(np.multiply(image[k:k+n, l:l+n, :], filt)) + bias[j]
                        out_arr[i][k][l][j] = res
        self.activation_val = out_arr
        return out_arr