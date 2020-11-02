import numpy as np
import matplotlib.pyplot as plt
from math import floor as math_floor
from lib.Utils.ActivationFunctions import *
from lib.Utils.CostFunctions import *


def initialize_weights(layers_dims, initialization="random"):
    """
    Initialize Weights and Bias for neural networks

    Arguments:
    layer_dims -- python array (list) containing the size of each layer including the input and the output.
    initialization -- method to initialize the data, stored as a text string
                      Choice: "random", "he"

    Returns:
    W -- python dictionary containing the weights of each layer
            1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
            ...
            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
    b -- python dictionary containing the bias of each layer
            1 -- weight matrix of shape (layers_dims[1], 1)
            ...
            WL -- weight matrix of shape (layers_dims[L], 1)
    """
    W = {}
    b = {}
    if initialization == "random":
        W, b = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        W, b = initialize_parameters_he(layers_dims)

    return W, b


def initialize_parameters_random(layers_dims):
    """
    Initialize randomly Weights and Bias for neural networks

    Arguments:
    layer_dims -- python array (list) containing the size of each layer including the input and the output.

    Returns:
    W -- python dictionary containing the weights of each layer
            1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
            ...
            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
    b -- python dictionary containing the bias of each layer
            1 -- weight matrix of shape (layers_dims[1], 1)
            ...
            WL -- weight matrix of shape (layers_dims[L], 1)
    """
    W = dict()
    b = dict()
    number_of_layers = len(layers_dims)  # integer representing the number of layers

    np.random.seed(7)
    for i_layer in range(1, number_of_layers):
        W[i_layer] = np.random.randn(layers_dims[i_layer], layers_dims[i_layer - 1]) * 0.01
        b[i_layer] = np.zeros((layers_dims[i_layer], 1))

    return W, b


def initialize_parameters_he(layers_dims):
    """
    Initialize Weights and Bias for neural networks using the He Initialization
    This is named for the first author of He et al., 2015.
    This is quite similar to Xavier initialization
    W_l = sqrt(2./layers_dims[l-1]

    Arguments:
    layer_dims -- python array (list) containing the size of each layer including the input and the output.

    Returns:
    W -- python dictionary containing the weights of each layer
            1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
            ...
            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
    b -- python dictionary containing the bias of each layer
            1 -- weight matrix of shape (layers_dims[1], 1)
            ...
            WL -- weight matrix of shape (layers_dims[L], 1)
    """
    W = dict()
    b = dict()
    number_of_layers = len(layers_dims)  # integer representing the number of layers

    for i_layer in range(1, number_of_layers):
        W[i_layer] = np.random.randn(layers_dims[i_layer], layers_dims[i_layer - 1]) * np.sqrt(2 / layers_dims[i_layer - 1])
        b[i_layer] = np.zeros((layers_dims[i_layer], 1))

    return W, b


def dropout_forward(A, keep_prob=1.):
    """
    Implement the forward (propagation) dropout
    Randomly shuts down some neurons (in the matrix A)

    Arguments:
    A -- numpy array corresponding to the output of neurons which are going to be shut down
    keep_prob -- probability of keeping a neuron active during drop-out, scalar

    Returns:
    A_dropout -- Output of neurons (or matrix A) after drop-out, numpy array with the same size as A
    D -- numpy array indicating which neurons have been shut down, same size as A_dropout
    """

    D = np.random.rand(A.shape[0], A.shape[1])  # Initialize matrix
    D = (D < keep_prob).astype(int)  # Convert to 0 or 1
    A_dropout = A * D / keep_prob  # Shut down some neurons and scale the value

    return A_dropout, D


def dropout_backward(A, D, keep_prob=1.):
    """
    Implement the backward (propagation) dropout

    Arguments:
    A -- numpy array corresponding to the output of neurons which are going to be shut down
    D -- numpy array indicating which neurons have been shut down, same size as A_dropout
    keep_prob -- probability of keeping a neuron active during drop-out, scalar

    Returns:
    A_dropout -- Output of neurons (or matrix A) after drop-out, numpy array with the same size as A
    """

    A_dropout = A * D
    A_dropout = A_dropout / keep_prob

    return A_dropout


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random mini-batches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true output vector, of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" mini-batches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math_floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def update_parameters_with_gd(W, b, grads, learning_rate):
    """
    Update parameters (Weight 'W', bias 'b') using gradient descent
    W = W - lr * dW
    b = b - lr * db

    Arguments:
    W -- python dictionary containing the weights:
                    W[1] for weights of the layer 1
                    W[2] for weights of the layer 2...
    b -- python dictionary containing the bias:
                    b[1] for weights of the layer 1
                    b[2] for weights of the layer 2...
    grads -- python dictionary containing the current gradients for each parameters:
                    grads['W' + str(l)] = dWl
                    grads['b' + str(l)] = dbl
    learning_rate -- learning rate for the gradient descent, scalar

    Returns:
    W -- python dictionary containing the updated Weights
    b -- python dictionary containing the updated bias
    """

    nb_layers = len(W)

    for i_layer in range(nb_layers):
        W[i_layer + 1] = W[i_layer + 1] - learning_rate * grads["W" + str(i_layer + 1)]
        b[i_layer + 1] = b[i_layer + 1] - learning_rate * grads["b" + str(i_layer + 1)]

    return W, b


def update_parameters_with_momentum(W, b, grads, velocity, learning_rate, beta):
    """
    Update parameters (Weight 'W', bias 'b') using Momentum
    velocity = beta * velocity + (1 - beta) * grads
    W = W - lr * velocity
    b = b - lr * velocity

    Arguments:
    W -- python dictionary containing the weights:
                    W[1] for weights of the layer 1
                    W[2] for weights of the layer 2...
    b -- python dictionary containing the bias:
                    b[1] for weights of the layer 1
                    b[2] for weights of the layer 2...
    grads -- python dictionary containing the current gradients for each parameters:
                    grads['W' + str(l)] = dWl
                    grads['b' + str(l)] = dbl
    velocity -- python dictionary containing the current velocity:
                    velocity['W' + str(l)] = ...
                    velocity['b' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    W -- python dictionary containing the updated Weights
    b -- python dictionary containing the updated bias
    velocity -- python dictionary containing the updated velocities
    """

    nb_layers = len(W)

    # Momentum update for each parameter
    for i_layer in range(nb_layers):
        # compute velocities
        velocity["W" + str(i_layer + 1)] = beta * velocity['W' + str(i_layer + 1)] + (1 - beta) * grads['W' + str(i_layer + 1)]
        velocity["b" + str(i_layer + 1)] = beta * velocity['b' + str(i_layer + 1)] + (1 - beta) * grads['b' + str(i_layer + 1)]

        # update parameters
        W[i_layer + 1] = W[i_layer + 1] - learning_rate * velocity["W" + str(i_layer + 1)]
        b[i_layer + 1] = b[i_layer + 1] - learning_rate * velocity["b" + str(i_layer + 1)]

    return W, b, velocity


def update_parameters_with_adam(W, b, grads, v, s, adam_counter, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters (Weight 'W', bias 'b') using Adam

    Arguments:
    W -- python dictionary containing the weights:
                    W[1] for weights of the layer 1
                    W[2] for weights of the layer 2...
    b -- python dictionary containing the bias:
                    b[1] for weights of the layer 1
                    b[2] for weights of the layer 2...
    grads -- python dictionary containing the current gradients for each parameters:
                    grads['W' + str(l)] = dWl
                    grads['b' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    adam_counter -- counts the number of steps taken of Adam
    learning_rate -- the learning rate, scalar
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    W -- python dictionary containing the updated Weights
    b -- python dictionary containing the updated bias
    v_corrected -- Adam variable, moving average of the first gradient, python dictionary
    s_corrected -- Adam variable, moving average of the squared gradient, python dictionary
    """

    np.seterr(invalid='raise')
    nb_layers = len(W)  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for i_layer in range(nb_layers):
        # Moving average of the gradients
        v["W" + str(i_layer + 1)] = beta1 * v["W" + str(i_layer + 1)] + (1 - beta1) * grads["W" + str(i_layer + 1)]
        v["b" + str(i_layer + 1)] = beta1 * v["b" + str(i_layer + 1)] + (1 - beta1) * grads["b" + str(i_layer + 1)]

        # Compute bias-corrected first moment estimate
        v_corrected["W" + str(i_layer + 1)] = v["W" + str(i_layer + 1)] / (1 - np.power(beta1, adam_counter))
        v_corrected["b" + str(i_layer + 1)] = v["b" + str(i_layer + 1)] / (1 - np.power(beta1, adam_counter))

        # Moving average of the squared gradients
        s["W" + str(i_layer + 1)] = beta2 * s["W" + str(i_layer + 1)] + (1 - beta2) * np.power(grads["W" + str(i_layer + 1)], 2)
        s["b" + str(i_layer + 1)] = beta2 * s["b" + str(i_layer + 1)] + (1 - beta2) * np.power(grads["b" + str(i_layer + 1)], 2)

        # Compute bias-corrected second raw moment estimate
        s_corrected["W" + str(i_layer + 1)] = s["W" + str(i_layer + 1)] / (1 - np.power(beta2, adam_counter))
        s_corrected["b" + str(i_layer + 1)] = s["b" + str(i_layer + 1)] / (1 - np.power(beta2, adam_counter))

        # Update parameters
        W[i_layer + 1] = W[i_layer + 1] - learning_rate * v_corrected["W" + str(i_layer + 1)] / (
                         np.sqrt(s_corrected["W" + str(i_layer + 1)]) + epsilon)
        b[i_layer + 1] = b[i_layer + 1] - learning_rate * v_corrected["b" + str(i_layer + 1)] / (
                         np.sqrt(s_corrected["b" + str(i_layer + 1)]) + epsilon)

    return W, b, v_corrected, s_corrected


class MultiLayerPerceptron:

    def __init__(self, layers_dims, initialization="he"):
        self.activationFunctionOutput = sigmoid
        self.activationFunctionHidden = relu
        self.costFunction = binary_cross_entropy
        self.activationFunctionOutput_grad = eval(self.activationFunctionOutput.__name__ + '_grad')
        self.activationFunctionHidden_grad = eval(self.activationFunctionHidden.__name__ + '_grad')
        self.costFunction_grad = eval(self.costFunction.__name__ + '_grad')

        self.layers_dims = layers_dims
        self.nb_layers = len(layers_dims) - 1  # number of layers in the neural network (less the input layer)

        # Dictionaries for the neural network parameters (Weight and bias)
        self.W, self.b = initialize_weights(layers_dims, initialization)

        # Dictionaries for the internal computations for the back-propagation
        self.A = dict()
        self.Z = dict()

        # L2 regularization
        self.regularization_lambda = 0.

        # Drop out
        self.keep_prob = 1.
        self.D = dict()  # Dictionary for the dropout feature

        # Optimizers
        self.V = dict()
        self.S = dict()
        self.beta = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon =1e-8
        self.adam_counter = 0  # Counter for Adam optimizer

    def forward_one_layer(self, layer, activation_function):
        """
        Implement the forward propagation for one layer
        A = f(Z)
        Z = A_prev * W + b

        Arguments:
        layer -- Layer to apply the forward propagation, scalar
        activation_function -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function
        """

        Z = np.dot(self.W[layer], self.A[layer - 1]) + self.b[layer]
        A = activation_function(Z)
        self.Z[layer] = Z

        assert (A.shape == (self.W[layer].shape[0], self.A[layer - 1].shape[1]))

        return A

    def forward_propagation(self, X):
        """
        Implement forward propagation for all layers of the Neural Network

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        A -- last post-activation value
        """

        self.A[0] = X
        number_of_layers = self.nb_layers

        # Implement forward propagation for the HIDDEN layers
        # A = Act_func(W_L * A_prev + b_L)
        for i_layer in range(number_of_layers-1):
            A = self.forward_one_layer(i_layer + 1, activation_function=self.activationFunctionHidden)
            self.A[i_layer + 1], self.D[i_layer + 1] = dropout_forward(A, self.keep_prob)

        # Implement forward propagation for the OUTPUT layer
        self.A[number_of_layers] = self.forward_one_layer(number_of_layers,
                                                          activation_function=self.activationFunctionOutput)

        assert (self.A[number_of_layers].shape == (1, X.shape[1]))

        return self.A[number_of_layers]

    def compute_loss(self, estimated_val, actual_val):
        """
        Implement the cost function

        Arguments:
        estimated_val -- probability vector corresponding to the predicted outputs, shape (1, number of examples)
        actual_val -- true "output" vector, shape (1, number of examples)

        Returns:
        cost -- cost value
        """

        m = actual_val.shape[1]

        # Compute loss from actual_val and estimated_val.
        cost = self.costFunction(actual_val, estimated_val)

        # Compute L2 regularization
        sum_weights = 0.
        for W in self.W.values():
            sum_weights = sum_weights + np.sum(np.square(W))
        l2_regularization_cost = self.regularization_lambda * sum_weights / (2 * m)

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        # Apply regularization
        cost = cost + l2_regularization_cost
        assert (cost.shape == ())

        return cost

    def backward_one_layer(self, dA, layer, activation_grad):
        """
        Implement the backward propagation for one layer

        Arguments:
        dA -- post-activation gradient for current layer
        layer -- layer to apply the backpropagation, scalar
        activation_grad -- the activation to be used in this layer, stored as a function

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev of the same layer
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W of the same layer
        db -- Gradient of the cost with respect to b (current layer l), same shape as b of the same layer
        """

        m = dA.shape[1]

        dZ = dA * activation_grad(self.Z[layer])
        dW = np.dot(dZ, self.A[layer-1].T) / m + self.regularization_lambda / m * self.W[layer]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W[layer].T, dZ)

        assert (dW.shape == self.W[layer].shape)
        assert (db.shape == (self.W[layer].shape[0], 1))
        assert (dA_prev.shape == self.A[layer-1].shape)

        return dA_prev, dW, db

    def backward_propagation(self, prediction, Y):
        """
        Implement the backward propagation for all layers of the Neural Network

        Arguments:
        prediction -- estimated vector, output of the forward propagation
        Y -- true "output" vector

        Returns:
        grads -- A dictionary with the gradients
                 grads["W" + str(l)] = ...
                 grads["b" + str(l)] = ...
                 grads["error" + str(l)] = ... Error of each neurons (also named delta)
        """
        grads = {}
        errors = {}
        nb_layers = self.nb_layers  # the number of layers
        Y = Y.reshape(prediction.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = self.costFunction_grad(Y, prediction)

        # Last layer with an unique activation function
        delta, grad_W, grad_b = self.backward_one_layer(dAL, nb_layers, self.activationFunctionOutput_grad)
        delta = dropout_backward(delta, self.D[nb_layers-1], self.keep_prob)  # Apply dropout
        grads["W" + str(nb_layers)] = grad_W
        grads["b" + str(nb_layers)] = grad_b
        grads["error" + str(nb_layers)] = delta

        # Loop from l=nb_layers to l=1
        for i_layer in reversed(range(1, nb_layers)):
            delta, grad_W, grad_b = self.backward_one_layer(delta, i_layer, self.activationFunctionHidden_grad)
            if i_layer > 1:
                delta = dropout_backward(delta, self.D[i_layer-1], self.keep_prob)  # Apply dropout
            grads["W" + str(i_layer)] = grad_W
            grads["b" + str(i_layer)] = grad_b
            grads["error" + str(i_layer)] = delta

        return grads, errors

    def initialize_optimizer(self, beta, beta1, beta2, epsilon, optimizer="gd"):

        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        if optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif optimizer == "momentum":
            for i_layer in range(self.nb_layers):
                self.V["W" + str(i_layer + 1)] = np.zeros(self.W[i_layer + 1].shape)
                self.V["b" + str(i_layer + 1)] = np.zeros(self.b[i_layer + 1].shape)
        elif optimizer == "adam":
            for i_layer in range(self.nb_layers):
                self.V["W" + str(i_layer + 1)] = np.zeros(self.W[i_layer + 1].shape)
                self.V["b" + str(i_layer + 1)] = np.zeros(self.b[i_layer + 1].shape)
                self.S["W" + str(i_layer + 1)] = np.zeros(self.W[i_layer + 1].shape)
                self.S["b" + str(i_layer + 1)] = np.zeros(self.b[i_layer + 1].shape)

    def update_parameters(self, grads, learning_rate, beta, beta1, beta2, epsilon, optimizer="gd"):

        # Update parameters
        if optimizer == "gd":
            self.W, self.b = update_parameters_with_gd(self.W, self.b, grads,
                                                       learning_rate=learning_rate)
        elif optimizer == "momentum":
            self.W, self.b, self.V = update_parameters_with_momentum(self.W, self.b, grads, self.V,
                                                                     learning_rate=learning_rate, beta=beta)
        elif optimizer == "adam":
            self.adam_counter += 1  # Adam counter
            self.W, self.b, self.V, self.S = update_parameters_with_adam(self.W, self.b, grads, self.V, self.S,
                                                                         learning_rate=learning_rate,
                                                                         beta1=beta1, beta2=beta2,
                                                                         adam_counter=self.adam_counter, epsilon=epsilon)

    def train(self, X, Y,
              mini_batch_size=64, num_epochs=10000, lambd=0.01, keep_prob=1.,
              learning_rate=0.01, optimizer="gd",
              beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
              print_cost=True, display=False):
        """
        Fit the input data X to the output data Y by learning

        Arguments:
        X -- input data, of shape (number of feature, number of examples)
        Y -- true "output" vector of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent, scalar
        num_epochs -- number of epochs, integer
        mini_batch_size -- the size of a mini batch, integer (use a power of 2)
        lambd -- regularization hyper-parameter (L2 regularization), scalar
        keep_prob -- probability of keeping a neuron active during drop-out, scalar
        print_cost -- if True, print the cost every 1000 iterations
        display -- if True, display the cost into a graph

        Returns:
        W -- Weight learnt by the model
        b -- Bias learnt by the model
        """

        costs = []
        seed = 1
        self.adam_counter = 0
        self.regularization_lambda = lambd
        self.keep_prob = keep_prob

        # Initizalize the optimizer
        self.initialize_optimizer(optimizer=optimizer, beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon)

        # Optimization loop
        for i in range(num_epochs):

            # Define the random mini-batches.
            # The seed is incremented to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                a_output = self.forward_propagation(minibatch_X)

                # Loss
                cost_total += self.compute_loss(a_output, minibatch_Y)

                # Backward propagation
                grads, _ = self.backward_propagation(a_output, minibatch_Y)

                # Update parameters
                self.update_parameters(grads, learning_rate, optimizer=optimizer,
                                       beta=self.beta, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)

            cost_avg = cost_total / X.shape[1]

            # Print the loss every 100 iterations
            if print_cost and (i % 1000 == 0 or i == num_epochs-1):
                print("Cost after iteration {}: {}".format(i, cost_avg))
            if print_cost and (i % 100 == 0 or i == num_epochs-1):
                costs.append(cost_avg)

        # plot the loss
        if display:
            plt.plot(costs)
            plt.ylabel('Cost')
            plt.xlabel('Iterations (per 100)')
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()

        return self.W, self.b, costs

    def predict(self, X, thresholds=None):
        """
        This function is used to predict the results of the trained neural network

        Arguments:
        X -- data set of examples you would like to predict, numpy array of shape (input size, number of examples)
        thresholds -- list of thresholds to label the output

        Returns:
        predictions -- predictions for the given dataset X
        """

        m = X.shape[1]
        predictions = np.zeros((1, m))

        # Disable dropout during prediction
        keep_prob = self.keep_prob
        self.keep_prob = 1.

        # Forward propagation
        estimated_val = self.forward_propagation(X)

        # Enable dropout after prediction
        self.keep_prob = keep_prob

        # convert estimated value to label
        if thresholds is not None:
            for i in range(0, len(thresholds)):
                predictions[0, estimated_val[0] >= thresholds[i]] = i+1

        return predictions

    def set_functions(self, activation_function_output=sigmoid, activation_function_hidden=relu, cost_function=mse):
        """
        This function allows the modification of one function (activation output, activation hidden, cost)

        Arguments:
        activation_function_output -- function of the activation function for the output layer (f(x))
        activation_function_hidden -- function of the activation function for the hidden layer(s) (f(x))
        cost_function -- function to compute the cost function (f(actual, estimated))

        """
        self.costFunction = cost_function
        self.activationFunctionOutput = activation_function_output
        self.activationFunctionHidden = activation_function_hidden


