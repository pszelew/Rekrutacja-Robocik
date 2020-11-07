from conv import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, input_dims: tuple, output_dims: tuple, n: int = 2):
        """
        n: int
            Size of window (n x n) 
        """
        super.__init__("max_pooling", input_dims, output_dims)
        self.n = n
    def forward(self, arr: np.array):
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
        self.activation_val = out_arr
        return out_arr