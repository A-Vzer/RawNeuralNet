# Amaan Valiuddin
import numpy as np
from keras.datasets import mnist
import random

# Raw neural network, with help of Coursera course of Andrew Ng and
# https://towardsdatascience.com/how-to-build-a-deep-neural-network-without-a-framework-5d46067754d5


# Labels are 0-9, this function changes it to one hot encoding
def to_categorical(labels):
    categorical = np.zeros((labels.shape[0], 10))
    for index, label in enumerate(labels):
        categorical[index, label] = 1
    return categorical


# If 0, all data is used. Otherwise 1, 10 or 100 images per class
images_per_class = 0
epochs = 60000
learning_rate = 0.0001

# Test train split
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:9000].reshape((784, 9000))
x_test = x_test[:1000].reshape((784, 1000))
y_train_vector = y_train[:9000]
y_train = to_categorical(y_train[:9000]).T
y_test = to_categorical(y_test[:1000]).T

digit_dict = {}
for i in range(10):
    idx = (y_train_vector == i)
    digit_dict[i] = x_train[idx]


# Define architecture
neural_net = [{"layer_size": 784, "activation": "none"},
              {"layer_size": 128, "activation": "relu"},
              {"layer_size": 10, "activation": "sig"}]


# Define dictionary with all weights, these are randomly initialized and multiplied by 0.001 to start with smaller
# weights
def parameter_initialization(neural_net):
    parameters = {}
    number_layers = len(neural_net)

    for l in range(1, number_layers):
        parameters['W' + str(l)] = np.random.randn(neural_net[l]["layer_size"], neural_net[l - 1]["layer_size"]) * 0.001
        parameters['b' + str(l)] = np.zeros(shape=(neural_net[l]["layer_size"], 1))

        assert(parameters['W' + str(l)].shape == (neural_net[l]["layer_size"], neural_net[l - 1]["layer_size"]))
        assert(parameters['b' + str(l)].shape == (neural_net[l]["layer_size"], 1))

    return parameters


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# Derivative of activation functions
def sigmoid_backward(dA, Z):
    dS = sigmoid(Z) * (1 - sigmoid(Z))
    return dA * dS


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# Forward propagation defined
def model_layer_forward(X, parameters, architecture):
    caches = {}
    A = X
    layer_numbers = len(architecture)
    caches['A' + str(0)] = A

    for l in range(1, layer_numbers):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        activation = neural_net[l]["activation"]
        Z, A = activation_forward_prop(A_prev, W, b, activation)
        caches['Z' + str(l)] = Z
        caches['A' + str(l)] = A

    AL = A

    return AL, caches


# Activation function per layer
def activation_forward_prop(A_prev, W, b, activation):
    if activation == 'sig':
        Z = forward_prop(A_prev, W, b)
        A = sigmoid(Z)

    elif activation == 'relu':
        Z = forward_prop(A_prev, W, b)
        A = relu(Z)

    return Z, A


# Matrix multiplication in forward propagation
def forward_prop(A, W, b):
    Z = np.dot(W, A) + b

    return Z


# Loss function
def cost_function(final_a, Y):
    m = Y.shape[1]
    loss_function = np.multiply(np.log(final_a), Y) + np.multiply(1 - Y, np.log(1 - final_a))
    loss_function = - np.sum(loss_function) / m

    return loss_function


# Back propagation process (finding weights and updating)
def model_layer_back(AL, Y, parameters, caches, architecture):
    gradients = {}
    layer_numbers = len(neural_net)
    m = AL.shape[1]
    dAL = - (np.true_divide(Y, AL) - np.true_divide(1 - Y, 1 - AL))
    dA_prev = dAL

    for l in reversed(range(1, layer_numbers)):
        dA_now = dA_prev
        activation = architecture[l]["activation"]
        W_now = parameters['W' + str(l)]
        Z_now = caches['Z' + str(l)]
        A_prev = caches['A' + str(l-1)]
        dA_prev, dW_now, db_now = activation_back(dA_now, Z_now, A_prev, W_now, activation)

        gradients["dW" + str(l)] = dW_now
        gradients["db" + str(l)] = db_now

    return gradients


def activation_back(dA, Z, A_prev, W, activation):
    if activation == "relu":
        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = backprop(dZ, A_prev, W)

    elif activation == "sig":
        dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = backprop(dZ, A_prev, W)

    return dA_prev, dW, db


# This calculates the derivatives
def backprop(dZ, A_prev, W):
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# Update weights
def update(parameters, gradients, alpha):
    L = len(neural_net)

    for l in range(1, L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - alpha * gradients["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * gradients["db" + str(l)]
    return parameters


# alpha == learning rate
def implementation(X, Y, architecture, alpha=learning_rate, iterations=epochs, print_iter=True):
    parameters = parameter_initialization(architecture)

    # SGD implementation
    for k in range(0, iterations):
        rand = random.randint(0, X.shape[1] - 1)
        X_sgd = X[:, rand].reshape(784, 1)
        Y_sgd = Y[:, rand].reshape(10, 1)
        AL, cache = model_layer_forward(X_sgd, parameters, architecture)
        cost = cost_function(AL, Y_sgd)
        gradients = model_layer_back(AL, Y_sgd, parameters, cache, architecture)
        parameters = update(parameters, gradients, alpha)
        if print_iter and k % 100 == 0:
            print("Loss after iteration %i: %f" %(k, cost))

    return parameters


# Obtain predicted digit
y_true_test = np.argmax(y_test, axis=0)
y_true_train = np.argmax(y_train, axis=0)

# execute depending on images per class hyper parameter
if images_per_class == 0:
    x_sub_train = x_train
    y_sub_train = y_train

else:
    x_sub_train = np.zeros((784, images_per_class * 10))
    y_sub_train = np.zeros((images_per_class * 10))
    for i in range(10):
        x_sub_train[:, i * images_per_class:images_per_class * (i + 1)] = digit_dict[i][0:images_per_class].T
        y_sub_train[i * images_per_class:images_per_class * (i + 1)] = i

y_sub_train = to_categorical(y_sub_train).T
final_parameters = implementation(x_sub_train, y_sub_train, neural_net)
AL_train = model_layer_forward(x_sub_train, final_parameters, neural_net)
AL_test = model_layer_forward(x_test, final_parameters, neural_net)

y_hat_train = np.argmax(AL_train, axis=0)
y_hat_test = np.argmax(AL_test, axis=0)
accuracy_test = (y_hat_test == y_true_test).mean()
accuracy_train = (y_hat_train == y_true_train).mean()
print(f"Accuracy test = {accuracy_test} \n"
      f"Accuracy train = {accuracy_train}")