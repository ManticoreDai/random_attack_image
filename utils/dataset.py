import numpy as np

def get_mnist_data(only_test=True):
    from tensorflow import keras

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test_255 = x_test.copy()
    x_test_255 = np.expand_dims(x_test_255, -1)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if only_test:
        del x_train, y_train
        return (x_test, x_test_255)
    else:
        return (x_test, x_test_255), x_train
