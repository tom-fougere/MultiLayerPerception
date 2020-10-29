import scipy.io
from NeuralNetwork import MultiLayerPerceptron
import matplotlib.pyplot as plt
import numpy as np

DISPLAY_DATA = False


def test_xor():

    mlp_xor = MultiLayerPerceptron([2, 3, 1])
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float).transpose()
    Y = np.array([[0], [1], [1], [0]], dtype=float).transpose()

    _, _, costs = mlp_xor.train(X, Y, learning_rate=0.1, num_iterations=15000, display=False)
    assert(costs[-1] < costs[0])

    prediction = mlp_xor.predict(X, [0.5])
    np.testing.assert_equal(prediction, [[0, 1, 1, 0]])

    if DISPLAY_DATA:
        plt.figure()
        plot_decision_boundary(lambda x: mlp_xor.predict(x.T, [0.5]), x, y, margin=0.2)
        plt.show()


def test_regularization():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T

    # No regularization
    mlp_regul0 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs0 = mlp_regul0.train(X, Y, learning_rate=0.9, num_iterations=40000, lambd=0, display=False)

    mlp_regul1 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs1 = mlp_regul1.train(X, Y, learning_rate=0.9, num_iterations=40000, lambd=0.1, display=False)

    assert(costs1[-1] > costs0[-1])

    if DISPLAY_DATA:
        plt.figure()
        plt.subplot(121)
        plot_decision_boundary(lambda x: mlp_regul0.predict(x.T, [0.5]), X, Y, margin=0.1)
        plt.title("Model without regularization")

        plt.subplot(122)
        plot_decision_boundary(lambda x: mlp_regul1.predict(x.T, [0.5]), X, Y, margin=0.5)
        plt.title("Model with regularization")
        plt.show()


def test_dropout():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T

    # No regularization
    mlp_dropout0 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs0 = mlp_dropout0.train(X, Y, learning_rate=0.9, num_iterations=40000, keep_prob=1., display=False)

    # With regularization
    mlp_dropout1 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs1 = mlp_dropout1.train(X, Y, learning_rate=0.9, num_iterations=40000, keep_prob=0.8, display=False)

    assert(costs1[-1] > costs0[-1])

    if DISPLAY_DATA:
        plt.figure()
        plt.subplot(121)
        plot_decision_boundary(lambda x: mlp_dropout0.predict(x.T, [0.5]), X, Y, margin=0.1)
        plt.title("Model without regularization")

        plt.subplot(122)
        plot_decision_boundary(lambda x: mlp_dropout1.predict(x.T, [0.5]), X, Y, margin=0.1)
        plt.title("Model with regularization")
        plt.show()


def plot_decision_boundary(model, X, y, margin=1.):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - margin, X[0, :].max() + margin
    y_min, y_max = X[1, :].min() - margin, X[1, :].max() + margin
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], edgecolors="black", c=y, cmap=plt.cm.Spectral)
    plt.draw()


if __name__ == '__main__':
    test_dropout()


