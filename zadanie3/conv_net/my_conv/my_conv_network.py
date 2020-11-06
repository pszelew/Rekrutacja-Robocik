from __future__ import annotations
import numpy as np

# Load numpy library

print("Loading my_network_network")
print(f"Numpy version {np.__version__}")
# Print image to know that we loaded our network package

# Proposed network structure
# 1) conv2d(3x3)
# 2) max_polling(2x2)
# 3) conv2d(3x3)
# 4) max_polling(2x2)
# 5) conv2d(3x3)
# 6) flatten
# 7) dense(ReLU)
# 8) dense(softmax)


class MyConvNetwork:



def relu(val: np.float64) -> np.float64:
    """ReLU function

    Parameters
    ----------
    val: np.float64
        Input value of ReLU function

    Returns
    -------
    np.float64
        Result of ReLU function
    """
    if val > 0:
        return val
    return np.float64(0)

def softmax(arr: np.array) -> np.array:
    """Softmax function

    Parameters
    ----------
    arr: np.array
        Input array of softmax function

    Returns
    -------
    np.array
        Result of softmax function
    """
    temp_lst: list
    temp_lst = [] 
    exp_sum: np.float64
    exp_sum = np.sum([np.exp(item) for item in arr])
    #calculate sum(e^zi)
    for item in arr:
        # For every value in our given array
        temp_val: np.float64
        temp_val = (np.exp(item))/(exp_sum)
        temp_lst.append(temp_val)
    return np.array(temp_lst)

def max_polling(arr: np.array, n: int):
    """Max-pooling operation

    Parameters
    ----------
    arr: np.array
        Input array of max-pooling operation
    n: int
        Size of window (n x n) 
    Returns
    -------
    np.array
        Result of max-pooling operation
    """
    # print(arr.shape)
    temp_shape = arr.shape
    out_dims = (temp_shape[0], temp_shape[1]//n, temp_shape[2]//n, temp_shape[3])
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
                    out_arr[i][k][l][j] = np.max(slice[k*n:(k+1)*n, l*n:(l+1)*n])
    return out_arr
def flatten(arr: np.array) -> np.array:
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
    return np.ndarray.flatten(arr)



def Conv2d(data: np.array, weights: np.array, n: int):
    """Conv net operation

    Parameters
    ----------
    arr: np.array
        Input array of max-pooling operation
    n: int
        Size of window (n x n) 
    Returns
    -------
    np.array
        Result of max-pooling operation
    """
    # print(arr.shape)
    temp_shape = arr.shape
    out_dims = (temp_shape[0], temp_shape[1]//n, temp_shape[2]//n, temp_shape[3])
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
                    out_arr[i][k][l][j] = np.max(slice[k*n:(k+1)*n, l*n:(l+1)*n])
    return out_arr

if __name__ == "__main__":
    # To test our defined functions
    test_polling = np.array([[
        [[1], [2], [3], [4], [5], [6]],
        [[2], [3], [4], [5], [6], [7]],
        [[3], [4], [5], [6], [7], [8]],
        [[4], [5], [6], [7], [8], [9]],
        [[5], [6], [7], [8], [9], [10]],
        [[6], [7], [8], [9], [10], [11]]]]
    )
    print(test_polling.shape)
    print(type(relu(np.float64(1.3))))
    print(softmax(np.array([15, 0.4, 1])))
    print((max_polling(test_polling, 2)).shape)