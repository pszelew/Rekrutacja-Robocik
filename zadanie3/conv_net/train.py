from tensorflow.keras.datasets import mnist
from my_conv import my_conv_network
# Import mnist package from keras datasets. Its probablythe simplest way
import numpy as np

print(f"Numpy version: {np.__version__}")
# Import numpy to perform our calculations 


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Use mnist loader to load our data and split them to train and test data

print(f"Train images array shape: {train_images.shape}")
print(f"Test images array shape: {test_images.shape}")
print(f"Train labels array shape: {train_labels.shape}")
print(f"Test labels array shape: {test_labels.shape}")
# Print those messages to be able to easly debug program


my_conv_network.max_polling(np.expand_dims(train_images, axis=3), 2)