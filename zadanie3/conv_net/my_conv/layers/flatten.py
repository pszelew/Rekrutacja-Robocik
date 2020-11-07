from conv import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self, input_dims: tuple, output_dims: tuple):
        super.__init__("flatten", input_dims, output_dims)
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
        out_arr = np.ndarray.flatten(input)
        self.activation_val = out_arr
        return out_arr