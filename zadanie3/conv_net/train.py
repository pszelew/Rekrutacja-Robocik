import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# Dont show me info messages
from tensorflow.keras.datasets import mnist
# Import mnist
from tensorflow.keras.backend import one_hot
# Its helfull thing
from my_conv import my_conv_network
# Import mnist package from keras datasets. Its probablythe simplest way
import numpy as np
# Import Numpy
BATCH_SIZE = 8
EPOCHS = 2

print(f"Numpy version: {np.__version__}")
# Import numpy to perform our calculations 


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Use mnist loader to load our data and split them to train and test data

print(f"Train images array shape: {train_images.shape}")
print(f"Test images array shape: {test_images.shape}")
print(f"Train labels array shape: {train_labels.shape}")
print(f"Test labels array shape: {test_labels.shape}")
# Print those messages to be able to easly debug program

train_images = train_images.astype('float64') / 255
test_images = train_images.astype('float64') / 255

train_images = np.expand_dims(train_images, 3)
test_images = np.expand_dims(test_images, 3)

train_labels = one_hot(train_labels, 10)
test_labels = one_hot(test_labels, 10)
my_conv = my_conv_network.MyConvNetwork(batch_size=BATCH_SIZE)

num_batches:int
num_batches = train_images.shape[0]//BATCH_SIZE

for n in range(EPOCHS):
    print(f"Epoch: {n}")
    for i in range(num_batches):
        batch = train_images[8*i:8*(i+1)]
        batch_labels = test_labels[8*i:8*(i+1)]
        result = my_conv.forward(batch)
        print(f"Batch {i+1} out of {num_batches}")
        print(f"Loss: {np.mean(my_conv.crossentropy(result,  batch_labels))}")
        my_conv.backward( batch_labels)