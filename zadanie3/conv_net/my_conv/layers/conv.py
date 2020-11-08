from __future__ import annotations
from .layer import Layer
import numpy as np


class Conv(Layer):
    def __init__(self, activation_fcn: str, input_dims: tuple, output_dims: tuple):
        super().__init__("conv", input_dims, output_dims)
        act_dic = {"relu": self.relu}
        act_div_dic = {self.relu: self.relu_derivative}
        self.activation_fcn = act_dic[activation_fcn]
        self.activation_fcn_der = act_div_dic[self.activation_fcn] 
        
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
        self.input = input
        n = weights.shape[0]
        # Window size
        #print(input.shape)
        temp_shape = input.shape
        #print(f"Dlugosc rozmiary{weights.shape}")
        out_dims = self.output_dims
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
        self.output = self.relu(self.activation_val)
        return self.output
    def back(self, prop: np.array, prev_output: np.array, weights: np.array, bias: np.array, next_weights: np.array, lr: np.float64):
        # print("Start conv")
        loss_derivative: np.array
        loss_derivative = prop
        out_derivative: np.array
        activation_derivative: np.array
        # for k in range(loss_derivative.shape[0]):
        #     for i in range(loss_derivative.shape[1]):
        #         # For all neurons of current layer
        #         for j in range(prop.shape[1]):
        #             # Loss propagating from next layer
        #             loss_derivative[k][i] += next_weights[j][i] * prop[k][j]
        #             # Temp_der times weight connecting it to our neuron 


        out_derivative = self.activation_fcn_der(self.activation_val)
        #print(prop.shape)
        ## b) Calculate delta(lay_out)/delta(activation)

        activation_derivative = self.activation_derivative(weights, prev_output)
        # c) Calculate delta(activation)/delta(weights)

        prop = np.multiply(loss_derivative, out_derivative)
        #print(prop.shape)
        gradients = np.zeros((prev_output.shape[0], weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))
        #print(gradients.shape)
        for batch in range(gradients.shape[0]): #2 batch
            for x in range(activation_derivative.shape[1]): # 3 output
                for y in range(activation_derivative.shape[2]): # 3 output
                    for i in range(gradients.shape[1]): #3 w
                        for j in range(gradients.shape[2]): #3 w
                            for k in range(gradients.shape[3]): #64 w
                                for channel in range(gradients.shape[3]): #64 channel
                                    gradients[batch][i][j][k][channel] += activation_derivative[batch][x][y][i][j][k][channel] * prop[batch][x][y][channel]


        arr = np.zeros(self.input_dims)


        for batch in range(arr.shape[0]):
            for i in range(prop.shape[1]): # Move right in prop
                for j in range(prop.shape[2]): # Move left in prop
                    for channel in range(prop.shape[3]):
                        arr[batch, i:i+3, j:j+3, :] += prop[batch][i][j][channel] *  weights[..., channel]

        prop = arr
        return gradients, prop




    def relu(self, arr: np.array) -> np.array:
        """ReLU function

        Parameters
        ----------
        arr: np.array
            Input array of ReLU function

        Returns
        -------
        np.float64
            Result of ReLU function
        """
        out_arr: np.array
        out_arr = np.zeros_like(arr)
        for batch in range(arr.shape[0]):
            for channel in range(arr.shape[-1]):
                for i in range(arr.shape[1]):
                    for j in range(arr.shape[2]):
                        if arr[batch][i][j][channel] > 0:
                            out_arr[batch][i][j][channel] = arr[batch][i][j][channel]
                        else:
                            out_arr[batch][i][j][channel] = 0
        return out_arr
    def relu_derivative(self, arr: np.array) -> np.array:
        """ReLU derivative function

        Parameters
        ----------
        arr: np.array
            Input array of ReLU function

        Returns
        -------
        np.float64
            Result of ReLU derivative function
        """
        out_arr: np.array
        out_arr = np.zeros_like(arr)

        for batch in range(arr.shape[0]):
            for channel in range(arr.shape[-1]):
                for i in range(arr.shape[1]):
                    for j in range(arr.shape[2]):
                        if out_arr[batch][i][j][channel] > 0:
                            out_arr[batch][i][j][channel] = 1
                        else:
                            out_arr[batch][i][j][channel] = 0
        return out_arr
    
    def activation_derivative(self, weights: np.array, last_output: np.array) -> np.array:
        """Derivative of activation funtion

        Parameters
        ----------
        weights: np.array
            Weights connecting previous layer with current
        last_output: np.array
            Output of last layer
        Returns
        -------
        np.array
            Derivative of activation function
        """
        out_arr: np.array
        out_arr = np.zeros((last_output.shape[0],  self.output_dims[1], self.output_dims[2], weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))

        # Empty array containing output of derivative 
        #print(out_arr.shape)
        for batch in range(out_arr.shape[0]):
            # Iterate through all batches
            for i in range(out_arr.shape[1]):
                # Iterate through first dim of output
                for j in range(out_arr.shape[2]):
                    # Second dim of output
                    for k in range(out_arr.shape[3]):
                        # First dim of filter
                        for l in range(out_arr.shape[4]):
                            # Second dim of filter
                                for m in range(out_arr.shape[5]):
                                # Third dim of filter
                                    for channel in range(out_arr.shape[6]):
                                        # Channels
                                        out_arr[batch][i][j][k][l][m][channel] = last_output[batch][k][l][m]
                                        #print(last_output.shape)
                                
        return out_arr
        # Return average of derivatives for every data in batch