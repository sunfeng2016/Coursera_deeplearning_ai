# -*- coding: utf-8 -*-

"""
Author: SunFeng
About:  Course1---Neural Networks and Deep Learning
        Week4---Deep Nueral Networks
        Assigment1---Building your Deep Neural Network: Step by Step
Date:   2019/7/29
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# Initialization

# 1. 2-layer Neural Network

def initialize_parameters(n_x, n_h, n_y):
    '''
    Initialize parameters for a two-layer network
    Arguments:
        @n_x: size of the input layer
        @n_h: size of the hidden layer
        @n_y: size of the output layer
    Returns:
        @parameters: python dictionary containing your parameters:
            @W1: weight matrix of shape (n_h, n_x)
            @b1: bias vector of shape (n_h, 1)
            @W2: weight matrix of shape (n_y, n_h)
            @b2: bias vector of shape (n_y, 1)
    '''

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# 2. L-layer Neural Network

def initialize_parameters_deep(layer_dims):
    
    '''
    Initailize parameters for an L-layer neural network
    Arguments:
        @layer_dims: python array (list) containing the dimensions of each layer in our network
    Returns:
        @parameters: python dictionary containing your parameters "W1", "b1",...
            @W1: weight matrix of shape (layer_dims[l], layer_dims[l-1])
            @b1: bias vector of shape (layer_dims[1], 1)
    '''

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)     # number of layers in the network

    for l in range(1, L):
        # Random Initialization
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

# Forward propagation module

# 1. Linear Forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation

    Arguments:
        @A: activations from previous layer (or input data) of shape (size of previous layer, number of examples)
        @W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
        @b: bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
        @z: the input of activation function, also called pre-activation parameter
        @cache: a python dictionary containing "A", "W", and "b"; stored for computing the backword pass efficiently
    """
    
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# 2. Linear-Activation Forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
        @A_prev: activations from previous layer (or input data) of shape (size of previous layer, number of examples)
        @W: weight matrix: numpy array of shape (size pf current layer, size of previous layer)
        @b: bias vector: numpy array of shape (size of the current layer, 1)
        @activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns: 
        @A: the output of the activation function, also called the post-activation value
        @cache: a python tuple containing "linear_cache" and "activation_cache"; stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":

        # Inputs: A_prev, W, b
        # Outputs: A, activation_cache
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: A_prev, W, b
        # Outputs: A, activation_cache
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# L-layer Model

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1) -> [LINEAR->SIGMOID] computation
    
    Arguments:
        @X -- data, numpy array of shape (input size, number of examples)
        @parameters -- output of initialize_parameters_deep()
    Returns:
        @AL -- last post-activation value
        @caches -- list of caches containing:
            every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
            the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X 
    L = len(parameters) // 2 # number of layers in the neural network

    # Implement [LINEAR -> RELU] * (L-1).
    # Add "cache" to the "caches" list
    for l in range(1, L):

        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Implement [LINEAR -> SIGMOID]
    # Add "cache" to the "caches" list
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

# Cost function
def compute_cost(Al, Y):
    """
    Implement the cost function (the cross-entropy cost J)
    
    Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape of (1, number of examples)

    Returns:
        cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y
    cost = - (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)) / m
    cost = np.squeeze(cost) # To make sure your cost's shape is waht we expect
    assert(cost.shape == ())

    return cost

# Backward propagation module

# 1. Linear backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns:
        dA -- Gradient of the cost with respect to the activation (of the prevous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to the W (current layer l), same shape as W
        db -- Gradient of the cost with respect to the b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

# 2. Linear-Activation backward
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer

    Arguments:
        dA -- post-activation gradient for current layer 1
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation function to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
        dA_prev -- Gradient of cost with respect to the activation (of the previous layer l - 1), same shape as A_prev
        dW -- Gradient of cost with respect to W (current layer l), same shape as W
        db -- Gradient of cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache

    if activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# 3. L-Model Backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backword propagaiton for the [LINEAR->RELU] * (L - 1) -> LINEAR -> SIGMOID group
    Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing: every cache of linear_activation_forward() with "relu" 
                  (it's caches[1], for l in range(L - 1) i.e l = 0...L-2)
                  the cache of linear_activation_foreward() with "sigmoid" (it's caches[L-1])
    Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1] # the number of examples
    Y = Y.reshape(AL.shape)

    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: AL, Y, caches
    # Outputs: grads["dAL"], grads["dWL"], grads["dbL"]

    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients
        # Inputs: grads["dA" + str(l + 2)], caches
        # outputs: grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)]

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# 4. Update Parameters

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward
        learning_rate -- learning rate of the gradient descent updatte rule
    Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

if __name__ == "__main__":
    
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
