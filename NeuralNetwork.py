import numpy as np
import matplotlib.pyplot as plt
from lib.Utils.ActivationFunctions import *


def initialize_weights(layers_dims, initialization):
    # Initialize parameters dictionary.
    if initialization == "random":
        W, b = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        W, b = initialize_parameters_he(layers_dims)

    return W, b


def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    W = dict()
    b = dict()
    number_of_layers = len(layers_dims)  # integer representing the number of layers

    for i_layer in range(1, number_of_layers):
        W[i_layer] = np.random.randn(layers_dims[i_layer], layers_dims[i_layer - 1]) * 0.01
        b[i_layer] = np.zeros((layers_dims[i_layer], 1))

    return W, b


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
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
        self.activationFunctionHidden = sigmoid
        self.activationFunctionOutput_grad = sigmoid_grad
        self.activationFunctionHidden_grad = sigmoid_grad
        # self.costFunction = @sigmoid

        self.layers_dims = layers_dims
        self.nb_layers = len(layers_dims) - 1  # number of layers in the neural network (less the input layer)

        self.W, self.b = initialize_weights(layers_dims, initialization)

        self.A = dict()
        self.Z = dict()

    def activation_forward(self, A_prev, W, b, activation_function):
        """
        Implement the forward propagation for the ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation_function -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A_prev) + b
        A = activation_function(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        return A, Z

    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        self.A[0] = X
        number_of_layers = self.nb_layers

        # Implement forward propagation for the HIDDEN layers
        # A = Act_func(W_L * A_prev + b_L)
        for i_layer in range(number_of_layers-1):
            self.A[i_layer + 1], self.Z[i_layer + 1] = self.activation_forward(self.A[i_layer],
                                                                               self.W[i_layer + 1],
                                                                               self.b[i_layer + 1],
                                                                               self.activationFunctionHidden)

        # Implement forward propagation for the OUTPUT layer
        self.A[number_of_layers], self.Z[number_of_layers] = self.activation_forward(self.A[number_of_layers-1],
                                                                                     self.W[number_of_layers],
                                                                                     self.b[number_of_layers],
                                                                                     self.activationFunctionOutput)

        assert (self.A[number_of_layers].shape == (1, X.shape[1]))

        return self.A[number_of_layers]

    def compute_loss(self, estimated_val, actual_val):
        """
        Implement the cost function defined by equation (7).
        Compute the cross-entropy cost
        - np.sum(Y * np.log(X) + (1-Y) * np.log(1 - X)) / m

        Arguments:
        estimated_val -- probability vector corresponding to your label predictions, shape (1, number of examples)
        actual_val -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = actual_val.shape[1]  # Number of training data

        # Compute loss from x and y.
        cost = - np.sum(actual_val * np.log(estimated_val) + (1 - actual_val) * np.log(1 - estimated_val)) / m
        # cost = np.sum((estimated_val - actual_val)**2) / m

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    def activation_backward(self, dA, A, W, A_prev, activation_grad):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        # m = A_prev.shape[1]
        m = dA.shape[1]

        d_act_func = activation_grad(A)
        dZ = dA * d_act_func
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dW.shape == W.shape)
        assert (db.shape == (W.shape[0], 1))
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    def backward_propagation(self, prediction, Y):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        nb_layers = self.nb_layers  # the number of layers
        m = prediction.shape[1]
        Y = Y.reshape(prediction.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, prediction) - np.divide(1 - Y, 1 - prediction))
        # dAL = 2 * (prediction - Y) / m

        # Last layer with an unique activation function
        delta, grad_W, grad_b = self.activation_backward(dAL,
                                                         self.A[nb_layers],
                                                         self.W[nb_layers],
                                                         self.A[nb_layers-1],
                                                         self.activationFunctionOutput_grad)
        grads["W" + str(nb_layers)] = grad_W
        grads["b" + str(nb_layers)] = grad_b

        # Loop from l=nb_layers to l=1
        for i_layer in reversed(range(1, nb_layers)):
            delta, grad_W, grad_b = self.activation_backward(delta,
                                                             self.A[i_layer],
                                                             self.W[i_layer],
                                                             self.A[i_layer-1],
                                                             self.activationFunctionHidden_grad)
            grads["W" + str(i_layer)] = grad_W
            grads["b" + str(i_layer)] = grad_b

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        nb_layers = self.nb_layers  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for i_layer in range(nb_layers):
            self.W[i_layer + 1] = self.W[i_layer + 1] - learning_rate * grads["W" + str(i_layer + 1)]
            self.b[i_layer + 1] = self.b[i_layer + 1] - learning_rate * grads["b" + str(i_layer + 1)]

    def train(self, X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations to run gradient descent
        print_cost -- if True, print the cost every 1000 iterations
        initialization -- flag to choose which initialization to use ("zeros","random" or "he")

        Returns:
        parameters -- parameters learnt by the model
        """

        costs = []

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation
            a_output = self.forward_propagation(X)

            # Loss
            cost = self.compute_loss(a_output, Y)

            # Backward propagation.
            grads = self.backward_propagation(a_output, Y)

            # Update parameters.
            self.update_parameters(grads, learning_rate)

            # Print the loss every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            #    costs.append(cost)
                costs.append(cost)

        # plot the loss
        # plt.plot(costs)
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per hundreds)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # return parameters



    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        return dA_prev, dW, db


if __name__ == '__main__':
    mlp = MultiLayerPerceptron([2, 3, 1])

    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float).transpose()
    y = np.array([0, 1, 1, 0], dtype=float, ndmin=2)
    z = mlp.train(x, y)

    print(z)
