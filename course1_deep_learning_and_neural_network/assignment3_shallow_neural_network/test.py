# -*- coding: utf-8 -*-

# 1 - Packages
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)      # set a seed so that the result are consisent

# 2 - Dataset

X, Y = load_planar_dataset()    # load dataset
# a numpy-array (matrix) X that contains your features (x1, x2)
# a numpy-array (vector) Y that contains your labels (red: 0, blue: 1)

'''
plt.scatter(X[0, :], X[1, :], s=40, c=Y.flatten(), cmap=plt.cm.Spectral)
plt.show()
'''
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

'''
print("The shape of X is: " + str(shape_X))
print("The shape of Y is: " + str(shape_Y))
print("I have m = %d training examples!" % (m))
'''

# 3 Simple Logistic Regression

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the dicision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.flatten())

plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print('\nAccuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) + '%' + "(percentage of correctly labelled datapoints)")
