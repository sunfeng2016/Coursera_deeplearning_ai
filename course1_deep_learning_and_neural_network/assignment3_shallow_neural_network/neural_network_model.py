# -*- coding: utf-8 -*-

'''
Author: SunFeng
About: course 1: neural network and deeplearning
       week 3: shallow neural network
       Programming Assignment: Planar data classfication with a hidden layer
Date: 2019/7/26
'''
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, tanh, load_planar_dataset, load_extra_datasets

np.random.seed(1)   # set a seed so that the resuls are consistent

# load dataset
def load_dataset():
    '''
    load planar dataset
    Arguments:
    Returns:
        @X: your dataset features
        @Y: your dataset labels
    '''
    X, Y = load_planar_dataset()    # load dataset
    
    return X, Y

# Defining the neural network structure

'''
n_x: the size of the input layer
n_h: the size of the hidden layer (set this to 4)
n_y: the size of the output layer
'''

def layer_sizes(X, Y):
    '''
    Defining the neural network structure
    Arguments:
        @x: input dataset of shape (input size, number of examples)
        @y: labels of shape (output size, number of examples)
    Returns:
        @n_x: the size of input layer
        @n_h: the size of hidden layer
        @n_y: the size of output layer
    '''
    
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

# Initialize the model's parameters

def initialize_parameters(n_x, n_h, n_y):
    '''
    Initialize the model's parameters
    Argument:
        @n_x: the size of the input layer
        @n_h: the size of the hidden layer
        @n_y: the size of the output layer
    Returns:
        @params: python dictionary containing your parameters:
            @W1: weight matrix of shape (n_h, n_x)
            @b1: bias vector of shape (n_h, 1)
            @W2: weight matrix of shape (n_y, n_h)
            @b2: bias vector of shape (n_y, 1)
    '''

    np.random.seed(2)   # set a seed so that the result are consisent

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

# Implement forward propagation

def forward_propagation(X, parameters):
    '''
    implement forward propagation
    Argument:
        @X: input data of size (n_x, m)
        @parameters: python dictionary containing your parameters
    Returns:
        @A2: The sigmoid output of the second activation
        @cache: a dictionary containing "Z1", "A1", "Z2" and "A2"
    '''
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement forward propagation to calculate A2
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# Implentment compute cost

def compute_cost(A2, Y, parameters):
    '''
    Computes the cross-entropy cost
    Arguments:
        @A2: The sigmoid output of the second activation, of shape(1, number of examples)
        @Y: "true" labels vector of shape (1, number of examples)
        @parameters: python dictionary containing your parameters W1, b1, W2, b2
    Returns:
        @cost: cross-entropy cost
    '''
    m = Y.shape[1]  #number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)     # make sure cost is the dimension we expect
                                # E.g. turns [[17]] into 17

    assert(isinstance(cost, float))

    return cost

# Implement the function backward propagration

def backward_propagation(parameters, cache, X, Y):
    '''
    Implement the backward propagation
    Arguments:
        @parameters: python dictionary containing out parameters (W1, b1, W2, b2)
        @cache: a dictionary containing "Z1", "A1", "Z2", "A2". 
        @X: input data of shape (2, number of examples)
        @Y: "true" labels vector of shape(1, number of examples)
    Returns:
        @grads: python dictionary containing your gradients with respect to different parameters
    '''

    m = X.shape[1]

    # retrieve W1 and W2 from the dictionary parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # retrieve A1 and A2 from dictionary "cache"
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# update parameters
def update_parameters(parameters, grads, learning_rate = 1.2):
    '''
    Update parameters using the gradient descent update rule
    Arguments:
        @parameters: python dictionary containing your parameters
        @grads: python dictionary containing your gradient with respect to different parameters
        @learning_rate: the learning rate used to update parameters
    Returns:
        @parameters: python dictionary containing your updated parameters
    '''
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each grident from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Merge all function into the nerual network model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    '''
    Build your neural network in nn_model
    Arguments:
        @X: dataset of shape (2, number of examples)
        @Y: labels of shape (1, number of exampless)
        @n_h: size of the hidden layer
        @num_iterations: Number of iterations in gradient descent loop
        @print_cost: if True, print the cost every 100 iterations
    Returns:
        @parameters: parameters learnt by the model. They can then be used to predict
    '''
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize paramters, then retrieve W1, b1, W2, b2.
    # Inputs: "n_x, n_h, n_y"
    # Outputs: " parameters(W1, b1, W2, b2)"
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. 
        # Inputs: "X, parameters". 
        # Outputs: "A2, cache"
        A2, cache = forward_propagation(X, parameters)

        # Cost Function.
        # Inputs: "A2, Y, parameters"
        # Output: "cost"
        cost = compute_cost(A2, Y, parameters)

        # Backward propagation.
        # Inputs: "parameters, cache, X, Y"
        # Outputs: "grads"
        grads = backward_propagation(parameters, cache, X, Y)

        # Update parameters by using gradient descent.
        # Inputs: "parameters, grads"
        # Outputs: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        # Print the cost every 1000 iterations:
        if print_cost and i % 1000 == 0:
            print("Cost after iterations %i: %f" %(i, cost))

    return parameters

# Use forward propagation to predict results
def predict(parameters, X):
    '''
    Using the learned parameters, predicts a class for each example in X

    Arguments: 
        @parameters: python dictionary containing your paramters
        @X: input data of size (n_x, m)
    Returns:
        @predictions: vector of predictions of our model (red: 0 / bule: 1)
    '''
    # Computes probabilities using forward propagation, and classfies to 0/1 using 0.5 as threshold
    A2 = forward_propagation(X, parameters)[0]
    predictions = (A2 > 0.5)

    return predictions

if __name__ == "__main__":
    
    X, Y = load_dataset()
    # Build a model with a n_h-dimensional hidden layer
    parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.flatten())
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    # Print accuracy
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # Tuning hidden layer size
    plt.figure(figsize = (16, 32))
    hidden_layer_sizes = [2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(2, 3, i + 1)
        plt.title('Hidden Layer of size %d' %n_h)
        parameters = nn_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.flatten())
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
   
    plt.show()