# -*- coding: utf-8 -*-

"""
Author: SunFeng
About:  Course1---Neural Networks and Deep Learning
        Week4---Deep Nueral Networks
        Assigment2---Deep Neural Network - Application
Date:   2019/7/31
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage 
from dnn_app_utils_v2 import *
from lr_utils import load_dataset

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# Dataset
def dataset_preprocess():
    """
    Implementing Data Set Preprocessing
    Arguments:
    Returns:
        train_x: your training set features
        train_y: your training set labels
        test_x: your test set features
        test_y: your test set labels
        classes: 
    """
    # load data set
    train_x_orig,  train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the  training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T # 参数-1使得剩下的维度变为1维
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    return train_x, train_y, test_x, test_y, classes

# Two-layer neural network

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID
    
    Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one 
    # of the functions you'd previously implemented

    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2, and b2
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR-> SIGMOID
        # Inputs: "X, W1, b1"
        # Outputs: "A1, cache1, A2, cache2"
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation
        # Inputs: dA2, cache2, cache1
        # Outputs: dA1, dW2, db2, dA0(not used), dW1, db1
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

# L-layer Neural Network
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a L-layer neural network: [LINEAR->RELU] * (L-1) -> LINEAR->SIGMOID

    Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1)
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict
    """

    np.random.seed(1)
    costs = []

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU] * (L-1) -> LINEAR ->SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration {} : {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

if __name__ == "__main__":
    
    train_x, train_y, test_x, test_y, classes = dataset_preprocess()

    '''
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1    
    parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)
    '''
    layers_dims = [12288, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

    print_mislabeled_images(classes, test_x, test_y, pred_test)


