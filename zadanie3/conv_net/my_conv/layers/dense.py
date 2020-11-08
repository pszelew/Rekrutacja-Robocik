from .layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, activation_fcn: str, input_dims: tuple, output_dims: tuple):
        super().__init__("dense", input_dims, output_dims)
        act_dic = {"relu": self.relu, "softmax": self.softmax}
        act_div_dic = {self.relu: self.relu_derivative, self.softmax: self.softmax_derivative}
        self.activation_fcn = act_dic[activation_fcn]
        self.activation_fcn_der = act_div_dic[self.activation_fcn] 
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
        self.input = input
        out_dims: tuple
        out_dims = (input.shape[0], weights.shape[0])
        out_arr: np.array
        out_arr = np.zeros(out_dims)
        
        for i, data in enumerate(input):
            print(weights.shape)
            print(data.shape)
            self.activation_val[i] = np.matmul(weights, data) + bias
        out_arr = self.activation_fcn(self.activation_val)
        self.output = out_arr
        
        return out_arr
    def activation_derivative(self, weights: np.array, last_output: np.array) -> np.array:
        """Derivative of activation funtion: matmul()+bias

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
        out_arr = np.zeros((last_output.shape[0], weights.shape[0], weights.shape[1]))
        # Empty array containing output of derivative 
        for k, data in enumerate(last_output):
            # Iterate through all batches
            for i in range(out_arr.shape[1]):
                for j in range(out_arr.shape[2]):
                    # For every field in our matrix of derivatives
                    out_arr[k][i][j] += data[j]
        return out_arr
        # Return average of derivatives for every data in batch
    def back_first(self, prev_output: np.array, weights: np.array, bias: np.array, train_labels: np.array, lr: np.float64):
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

        # 1) For the last layer:
        #    a) Calculate delta(loss)/delta(net_out)
        #    b) Calculate delta(net_out)/delta(activation)
        #    c) Calculate delta(activation)/delta(weights)
        #    d) Use chain rule to get our ouptput

        loss_derivative: np.array
        loss_derivative = self.crossentropy_derivative(self.output, train_labels)
        # First delta(loss)/delta(net_out)

        net_out_derivative: np.array
        net_out_derivative = self.activation_fcn_der(self.activation_val)
        # Second calculate delta(net_out)/delta(activation)

        net_activation_derivative = self.activation_derivative(weights, prev_output)
        # Third calculate delta(activation)/delta(weights)

        # Calculate gradients using chain rule
        gradients: np.array
        gradients = np.zeros_like(net_activation_derivative)
        prop: np.array
        prop = np.multiply(loss_derivative, net_out_derivative)
        # It will propagate further. Later we return this
        for k in range(gradients.shape[0]):
            for i in range(gradients.shape[1]):
                for j in range(gradients.shape[2]):
                    gradients[k][i][j] = net_activation_derivative[k][i][j] * prop[k][i] 
        weights = weights - lr*gradients
        # Update weights
        
        #bias TBD

        return prop
        
    def back(self, prop: np.array, prev_output: np.array, weights: np.array, bias: np.array, next_weights: np.array, lr: np.float64):
        loss_derivative: np.array
        loss_derivative = np.zeros((self.output_dims[0], self.output_dims[1]))
        out_derivative: np.array
        activation_derivative: np.array
        for k in range(loss_derivative.shape[0]):
            for i in range(loss_derivative.shape[1]):
                # For all neurons of current layer
                for j in range(prop.shape[1]):
                    # Loss propagating from next layer
                    loss_derivative[k][i] += next_weights[j][i] * prop[k][j]
                    # Temp_der times weight connecting it to our neuron 

        out_derivative = self.activation_fcn_der(self.activation_val)
        # b) Calculate delta(lay_out)/delta(activation)

        activation_derivative = self.activation_derivative(weights, prev_output)
        # c) Calculate delta(activation)/delta(weights)

        prop = np.multiply(loss_derivative, out_derivative)

        gradients = np.zeros_like(activation_derivative)
        for k in range(gradients.shape[0]):
            for i in range(gradients.shape[1]):
                for j in range(gradients.shape[2]):
                    gradients[k][i][j] = activation_derivative[k][i][j] * prop[k][i] 
        weights = weights - lr*gradients
        #print(weights[0][0])
        weights = weights - lr*gradients
        #print(weights[0][0])
        return prop

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
        for i, data in enumerate(arr):
            out_arr[i] = np.array([item if item > 0 else np.float64(0.0) for item in data])
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
        out_arr = np.zeros((arr.shape[0], arr.shape[1]))
        for i, data in enumerate(arr):
            out_arr[i] = np.array([np.float64(1.0) if item > 0 else np.float64(0.0) for item in data])
        return out_arr

    def softmax(self, arr: np.array) -> np.array:
        """softmax function for a batch of data

        Parameters
        ----------
        arr: np.array
            Input array of softmax function

        Returns
        -------
        np.array
            Result of softmax function
        """
        out_arr: np.array
        out_arr = np.zeros_like(arr) 
        exp_sum: np.float64
        #calculate sum(e^zi)
        for i, data in enumerate(arr):
            data:np.array
            data = data - data.max()
            exp_sum = np.sum([np.exp(item) for item in data])
            for j, value in enumerate(data):
            # For every value in our given array
                temp_val: np.float64
                temp_val = (np.exp(value))/(exp_sum)
                out_arr[i][j] = temp_val
        return out_arr

    def softmax_derivative(self, arr: np.array) -> np.array:
        """Returns derivative of softmax function

        Parameters
        ----------
        arr: np.array
            Input array of softmax function (batch_size, data_size)

        Returns
        -------
        np.array
            Result of softmax derivative function
        """
        exp_sum: np.float64
        out_arr: np.array
        out_arr = np.zeros((arr.shape[0], arr.shape[1]))

        for k, data in enumerate(arr):
            data = data - data.max()
            exp_sum = np.sum([np.exp(item) for item in data])
            out_arr[k] = np.array([(np.exp(data[i]) * (exp_sum-np.exp(data[i]))) / (exp_sum**2) for i in range(len(data))])
            # Sum derivatives from all images from our batch
        return out_arr
        # Return average derivative


    def crossentropy(self, net_out : np.array, gr_tru: np.array) -> np.float64:
        """Crossentropy loss function

        Parameters
        ----------
        net_out: np.array
            Output of last layer neurons (batch_size, net_out_size)
        ground truth: np.array
            Ground truth for our outputs (batch_size, net_out_size)
        Returns
        -------
        np.array
            Loss value
        """
        out_arr: np.array
        out_arr = np.zeros(net_out.shape[0])
        # Return [batch_size] array
        for i, data in enumerate(net_out):
            y = gr_tru[i]
            out_arr[i] = -1 * np.sum([y[i]*np.log10(data[i]) + ((1-y[i])*np.log10(1-data[i])) for i in range(len(data))])
        return out_arr

    def crossentropy_derivative(self, net_out: np.array, gr_tru: np.array) -> np.array:
        """Returns derivative of crossentropy function given net_output and ground truth

        Parameters
        ----------
        net_out: np.array
            Output of last layer neurons (batch_size, net_out_size)
        ground truth: np.array
            Ground truth for our outputs (batch_size, net_out_size)
        Returns
        -------
        np.array
            Derivative of crossentropy loss fcn -- delta(loss)/delta(net_out)
        """
        out_arr: np.array
        out_arr = np.zeros((net_out.shape[0], net_out.shape[1]))
        # print(gr_tru.shape)
        print(gr_tru.shape)

        for i, data in enumerate(net_out):
            y = gr_tru[i]
            out_arr[i] = -1*(np.divide(y, data)) + np.divide((1-y), (1-data))
        return out_arr