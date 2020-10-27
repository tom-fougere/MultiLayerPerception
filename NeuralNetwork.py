import numpy as np
import matplotlib.pyplot as plt
from lib.Utils.ActivationFunctions import *
from lib.Utils.CostFunctions import *


def initialize_weights(layers_dims, initialization):
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
    initialization -- method of initialize the data, stored as a text string
                      Choice: random, he

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
    initialization -- method of initialize the data, stored as a text string
                      Choice: random, he

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


class MultiLayerPerceptron:

    def __init__(self, layers_dims, initialization="random"):
        self.activationFunctionOutput = sigmoid
        self.activationFunctionHidden = relu
        self.costFunction = binary_cross_entropy
        self.activationFunctionOutput_grad = eval(self.activationFunctionOutput.__name__ + '_grad')
        self.activationFunctionHidden_grad = eval(self.activationFunctionHidden.__name__ + '_grad')
        self.costFunction_grad = eval(self.costFunction.__name__ + '_grad')

        self.layers_dims = layers_dims
        self.nb_layers = len(layers_dims) - 1  # number of layers in the neural network (less the input layer)

        self.regularization_lambda = 0

        self.W, self.b = initialize_weights(layers_dims, initialization)
        self.A = dict()
        self.Z = dict()

    def forward_one_layer(self, A_prev, W, b, activation_function):
        """
        Implement the forward propagation for one layer
        A = f(Z)
        Z = A_prev * W + b

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation_function -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function
        Z -- the output without the activation function
        """

        Z = np.dot(W, A_prev) + b
        A = activation_function(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        return A, Z

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
            self.A[i_layer + 1], self.Z[i_layer + 1] = self.forward_one_layer(self.A[i_layer],
                                                                              self.W[i_layer + 1],
                                                                              self.b[i_layer + 1],
                                                                              self.activationFunctionHidden)

        # Implement forward propagation for the OUTPUT layer
        self.A[number_of_layers], self.Z[number_of_layers] = self.forward_one_layer(self.A[number_of_layers-1],
                                                                                    self.W[number_of_layers],
                                                                                    self.b[number_of_layers],
                                                                                    self.activationFunctionOutput)

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
        layer -- layer to apply the backpropagation
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
                 grads["error" + str(l)] = ... Error of each neurones (also named delta)
        """
        grads = {}
        errors = {}
        nb_layers = self.nb_layers  # the number of layers
        Y = Y.reshape(prediction.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = self.costFunction_grad(Y, prediction)

        # Last layer with an unique activation function
        delta, grad_W, grad_b = self.backward_one_layer(dAL, nb_layers, self.activationFunctionOutput_grad)
        grads["W" + str(nb_layers)] = grad_W
        grads["b" + str(nb_layers)] = grad_b
        grads["error" + str(nb_layers)] = delta

        # Loop from l=nb_layers to l=1
        for i_layer in reversed(range(1, nb_layers)):
            delta, grad_W, grad_b = self.backward_one_layer(delta, i_layer, self.activationFunctionHidden_grad)
            grads["W" + str(i_layer)] = grad_W
            grads["b" + str(i_layer)] = grad_b
            grads["error" + str(i_layer)] = delta

        return grads, errors

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent
        W = W - lr * dW
        b = b - lr * db

        Arguments:
        grads -- python dictionary containing your gradients
        learning_rate -- learning rate for the gradient descent
        """

        nb_layers = self.nb_layers  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for i_layer in range(nb_layers):
            self.W[i_layer + 1] = self.W[i_layer + 1] - learning_rate * grads["W" + str(i_layer + 1)]
            self.b[i_layer + 1] = self.b[i_layer + 1] - learning_rate * grads["b" + str(i_layer + 1)]

    def train(self, X, Y, learning_rate=0.01, num_iterations=15000, lambd=0.01, print_cost=True, display=False):
        """
        Fit the input data X to the output data Y by learning

        Arguments:
        X -- input data, of shape (number of feature, number of examples)
        Y -- true "output" vector of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations to run gradient descent
        print_cost -- if True, print the cost every 1000 iterations
        display -- if True, display the cost into a graph

        Returns:
        W -- Weight learnt by the model
        b -- Bias learnt by the model
        """

        costs = []
        self.regularization_lambda = lambd

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation
            a_output = self.forward_propagation(X)

            # Loss
            cost = self.compute_loss(a_output, Y)

            # Backward propagation.
            grads, _ = self.backward_propagation(a_output, Y)

            # Update parameters.
            self.update_parameters(grads, learning_rate)

            # Print the loss every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                costs.append(cost)

        # plot the loss
        if display:
            plt.plot(costs)
            plt.ylabel('Cost')
            plt.xlabel('Iterations (per hundreds)')
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()

        return self.W, self.b

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

        # Forward propagation
        estimated_val = self.forward_propagation(X)

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


if __name__ == '__main__':

    mlp = MultiLayerPerceptron([2, 2, 1])
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float).transpose()
    y = np.array([[0], [1], [1], [0]], dtype=float).transpose()
    mlp.train(x, y, learning_rate=0.1, num_iterations=15000, display=True)
    prediction = mlp.predict(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).transpose(), [0.5])
    print(prediction)

