from __future__ import annotations
from .layers.layer import Layer
from .layers.conv import Conv
from .layers.dense import Dense
from .layers.flatten import Flatten
from .layers.max_pooling import MaxPooling
import numpy as np

# Load numpy library

print("Loading my_network_network")
print(f"Numpy version {np.__version__}")
# Print image to know that we loaded our network package

# Proposed network structure. Loss function = categorical_crossentropy
# 1) conv2d(3x3)
# 2) max_polling(2x2)
# 3) conv2d(3x3)
# 4) max_polling(2x2)
# 5) conv2d(3x3)
# 6) flatten
# 7) dense(ReLU)
# 8) dense(softmax)


class MyConvNetwork:
    """
    A class used to represent a convolutional network used to solve MNIST detection problem

    Attributes
    ----------
    batch_size: int
        Batch size
    layers: list[Layer]
        List of layers
    weights: list[np.array]
        List keeping weights used by consecutive layers of network
    states: list[np.array]
        List containing consecutive values generated by neurons
    Methods
    -------
    relu(val: np.float64) -> np.float64
        ReLU function
    softmax(arr: np.array) -> np.array:
        softmax function
    max_polling(arr: np.array, n: int) -> np.array
        max-polling operation
    flatten(arr: np.array) -> np.array
        Operation flattening given array to 1D array
    conv2d_forward(input: np.array, weights: np.array, bias: np.array) -> np.array:
        max-polling operation
    """
    def __init__(self, lr: float = 0.001, batch_size: int = 2):  
        self.lr: float
        self.lr = lr
        self.weights: list
        self.weights = []
        
        self.weights.append(np.random.normal(size=(3, 3, 1, 32))*np.sqrt(2/32))
        # Weights for 1st conv layer
        self.weights.append(np.random.normal(size=(3, 3, 32, 64))*np.sqrt(2/32))
        # Weights for 2nd conv layer
        self.weights.append(np.random.normal(size=(3, 3, 64, 64))*np.sqrt(2/64))
        # Weights for 3rd conv layer
        self.weights.append(np.random.normal(size=(64, 576))*np.sqrt(2/576))
        # Weights for 1nd dense layer
        self.weights.append(np.random.normal(size=(10, 64))*np.sqrt(2/64))
        # Weights for 2nd dense layer

        self.bias: list
        self.bias = []
        self.bias.append(np.zeros(32))
        # Bias for 1st conv layer
        self.bias.append(np.zeros(64))
        # Bias for 2nd conv layer
        self.bias.append(np.zeros(64))
        # Bias for 3nd conv layer
        self.bias.append(np.zeros(64))
        # Bias for 1st dense layer
        self.bias.append(np.zeros(10))
        # Bias for 2nd dense layer

        self.layers: list
        self.layers = []
        self.batch_size: int
        self.batch_size = batch_size
        self.layers.append(Conv("relu", (batch_size, 28, 28, 1), (batch_size, 26, 26, 32)))
        # First conv layer
        self.layers.append(MaxPooling((batch_size, 26, 26, 32), (batch_size, 13, 13, 32)))
        # Max pooling layer to reduce size
        self.layers.append(Conv("relu", (batch_size, 13, 13, 32), (batch_size, 11, 11, 64)))
        # Second conv layer
        self.layers.append(MaxPooling((batch_size, 11, 11, 64), (batch_size, 5, 5, 64)))
        # Max pooling layer to reduce size
        self.layers.append(Conv("relu", (batch_size, 5, 5, 64), (batch_size, 3, 3, 64)))
        # Third conv layer
        self.layers.append(Flatten((batch_size, 3, 3, 64), (batch_size, 576)))
        # Flatten layer to prepare data for Dense layers
        self.layers.append(Dense("relu", (batch_size, 576), (batch_size, 64)))
        # First dense layer
        self.layers.append(Dense("softmax", (batch_size, 64), (batch_size, 10)))
        # Second dense layer. Output of network


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
    
    def forward(self, input_data: np.array) -> np.array:
        """Forward pass function

        Parameters
        ----------
        input_data: np.array
            Input array of our network (batch_size, 28, 28, 1)

        Returns
        -------
        np.array
            Prediction of network
        """
        count_weights: int
        count_weights = 0
        temp_data: np.array
        temp_data = input_data
        for layer in self.layers:
            # print("Forward z warstwa: " + layer.name)
            if layer.name is "conv":
                temp_data = layer.forward(temp_data, self.weights[count_weights], self.bias[count_weights])
                count_weights += 1
            elif layer.name is "dense":
                temp_data = layer.forward(temp_data, self.weights[count_weights], self.bias[count_weights])
                count_weights += 1
            elif layer.name is "flatten":
                temp_data = layer.forward(temp_data)
                pass
            elif layer.name is "max_pooling":
                temp_data = layer.forward(temp_data)
                pass
            else:
                print("Layer not found!")
                exit()
            #print(f"Layer {layer.name} output: {layer.output.max()}")
        return temp_data
    
    def backward(self, train_labels):
        """Backward pass of our network

        Update weights 

        Parameters
        ----------
        input_data: np.array
            Input array of our network

        Returns
        -------
        np.array
            Prediction of network
        """
        prop: np.array
        prop = np.array([])
        c: int
        c = 0
        for x, layer in enumerate(self.layers[::-1]):
            if x is 0:
                gradients, prop = layer.back_first(self.layers[-1*(x+2)].output, self.weights[-1*(c+1)], self.bias[-1*(c+1)], train_labels)
                self.weights[-1*(c+1)] = self.weights[-1*(c+1)] - self.lr * np.mean(gradients, axis=0, dtype=np.float64)
                c += 1
            elif x is 7:
                gradients, prop = layer.back(prop,  self.layers[-1*(x+1)].input, self.weights[-1*(c+1)], self.bias[-1*(c+1)], train_labels)
                self.weights[-1*(c+1)] = self.weights[-1*(c+1)] - self.lr * np.mean(gradients, axis=0, dtype=np.float64)
                c += 1    
            else:
                if layer.name is "conv":
                    #print("fasfs")
                    #print(prop)
                    gradients, prop = layer.back(prop, self.layers[-1*(x+2)].output, self.weights[-1*(c+1)], self.bias[-1*(c+1)])
                    self.weights[-1*(c+1)] = self.weights[-1*(c+1)] - self.lr * np.mean(gradients, axis=0, dtype=np.float64)
                    c += 1
                elif layer.name is "dense":
                    gradients, prop = layer.back(prop, self.layers[-1*(x+2)].output, self.weights[-1*(c+1)], self.bias[-1*(c+1)], self.weights[-1*c])
                    self.weights[-1*(c+1)] = self.weights[-1*(c+1)] - self.lr * np.mean(gradients, axis=0, dtype=np.float64)
                    c += 1
                elif layer.name is "flatten":
                    #print(prop.max())
                    prop = layer.back(prop, self.layers[-1*(x+2)].output, self.weights[-1*(x)])
                    #print("flat")
                    #print(prop)
                    pass
                elif layer.name is "max_pooling":
                    prop = layer.back(prop, self.layers[-1*(x+2)].output)
                    pass
                else:
                    print("Layer not found!")
                    exit()
        return prop
        
        
if __name__ == "__main__":
    my_conv = MyConvNetwork()
    # To test our defined functions
    train_images = np.random.rand(2, 28, 28, 1)
    train_images = train_images/train_images.max()
    test_labels = np.array([[1,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0]])
    test_labels = test_labels/test_labels.max()

    for i in range(20):
        result = my_conv.forward(train_images)
        print(f"Loss: {my_conv.crossentropy(result, test_labels)}")
        my_conv.backward(test_labels)