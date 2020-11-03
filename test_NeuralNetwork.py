import scipy.io
from NeuralNetwork import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

DISPLAY_DATA = False


def test_xor():

    mlp_xor = MultiLayerPerceptron([2, 3, 1])
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float).transpose()
    Y = np.array([[0], [1], [1], [0]], dtype=float).transpose()

    _, _, costs = mlp_xor.train(X, Y, learning_rate=0.1, num_epochs=15000, mini_batch_size=X.shape[1], display=False)
    assert(costs[-1] < costs[0])

    prediction = mlp_xor.predict(X, [0.5])
    np.testing.assert_equal(prediction, [[0, 1, 1, 0]])

    if DISPLAY_DATA:
        plt.figure()
        plot_decision_boundary(lambda x: mlp_xor.predict(x.T, [0.5]), X, Y, margin=0.2)
        plt.show()


def test_data():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T

    mlp = MultiLayerPerceptron([2, 20, 3, 1])

    _, _, costs = mlp.train(X, Y, learning_rate=0.7, mini_batch_size=64,
                            display=False, optimizer="gd", num_epochs=10000)
    assert (costs[-1] < costs[0])

    if DISPLAY_DATA:
        plt.figure()
        plot_decision_boundary(lambda x: mlp.predict(x.T, [0.5]), X, Y, margin=0.2)
        plt.show()


def test_optimizer():

    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    X = train_X.T
    Y = train_Y.reshape((1, train_Y.shape[0]))

    mlp_gd = MultiLayerPerceptron([2, 5, 2, 1])
    _, _, costs_gd = mlp_gd.train(X, Y, learning_rate=0.0007, mini_batch_size=64,
                                  display=False, optimizer="gd",
                                  beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=8000)

    mlp_momentum = MultiLayerPerceptron([2, 5, 2, 1])
    _, _, costs_momentum = mlp_momentum.train(X, Y, learning_rate=0.0007, mini_batch_size=64,
                                              display=False, optimizer="momentum",
                                              beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=8000)

    mlp_adam = MultiLayerPerceptron([2, 5, 2, 1])
    _, _, costs_adam = mlp_adam.train(X, Y, learning_rate=0.0007, mini_batch_size=64,
                                      display=False, optimizer="adam",
                                      beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=8000)

    assert (costs_gd[-1] < costs_gd[0])
    assert (costs_momentum[-1] < costs_momentum[0])
    assert (costs_adam[-1] < costs_adam[0])
    assert (costs_adam[-1] < costs_momentum[-1] < costs_gd[0])

    if DISPLAY_DATA:
        plt.figure()
        plt.subplot(131)
        plot_decision_boundary(lambda x: mlp_gd.predict(x.T, [0.5]), X, Y, margin=0.2)
        plt.title('Gradient descent')
        plt.subplot(132)
        plot_decision_boundary(lambda x: mlp_momentum.predict(x.T, [0.5]), X, Y, margin=0.2)
        plt.title('Momentum')
        plt.subplot(133)
        plot_decision_boundary(lambda x: mlp_adam.predict(x.T, [0.5]), X, Y, margin=0.2)
        plt.title('Adam')
        plt.show()


def test_regularization():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T

    # No regularization
    mlp_regul0 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs0 = mlp_regul0.train(X, Y, learning_rate=0.9, num_epochs=40000, mini_batch_size=X.shape[1], lambd=0,
                                    display=False)

    mlp_regul1 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs1 = mlp_regul1.train(X, Y, learning_rate=0.9, num_epochs=40000, mini_batch_size=X.shape[1], lambd=0.1,
                                    display=False)

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

    # No dropout
    mlp_dropout0 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs0 = mlp_dropout0.train(X, Y, learning_rate=0.9, num_epochs=40000, mini_batch_size=X.shape[1],
                                      keep_prob=1., display=False)

    # With dropout
    mlp_dropout1 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs1 = mlp_dropout1.train(X, Y, learning_rate=0.9, num_epochs=40000, mini_batch_size=X.shape[1],
                                      keep_prob=0.8, display=False)

    assert(costs1[-1] > costs0[-1])

    if DISPLAY_DATA:
        plt.figure()
        plt.subplot(121)
        plot_decision_boundary(lambda x: mlp_dropout0.predict(x.T, [0.5]), X, Y, margin=0.1)
        plt.title("Model without dropout")

        plt.subplot(122)
        plot_decision_boundary(lambda x: mlp_dropout1.predict(x.T, [0.5]), X, Y, margin=0.1)
        plt.title("Model with dropout")
        plt.show()


def test_minibatch():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T

    # No mini-batch
    mlp_minibatch0 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs0 = mlp_minibatch0.train(X, Y, learning_rate=0.9, num_epochs=4000, mini_batch_size=X.shape[1],
                                        lambd=0, display=False)

    # Mini-batch = Nb_examples/2
    mlp_minibatch1 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs1 = mlp_minibatch1.train(X, Y, learning_rate=0.9, num_epochs=4000, mini_batch_size=2,
                                        lambd=0, display=False)

    # Mini-batch = 32
    mlp_minibatch2 = MultiLayerPerceptron([X.shape[0], 20, 3, 1])
    _, _, costs2 = mlp_minibatch2.train(X, Y, learning_rate=0.9, num_epochs=4000, mini_batch_size=32,
                                        lambd=0, display=False)

    mean0 = np.mean(costs0[-10:-1])
    mean1 = np.mean(costs1[-10:-1])
    mean2 = np.mean(costs2[-10:-1])
    std0 = np.std(costs0[-10:-1])
    std1 = np.std(costs1[-10:-1])
    std2 = np.std(costs2[-10:-1])

    assert (mean0 < mean2 < mean1)
    assert (std0 < std2 < std1)

    if DISPLAY_DATA:
        plt.figure()
        plt.plot(costs0)
        plt.plot(costs1)
        plt.plot(costs2)
        plt.title("Costs for Minibatch")
        plt.legend(('No Minibatch', 'Minibatch = 2', 'Minibatch = 32'))
        plt.show()


def test_minibatch_subfunction():
    data = scipy.io.loadmat('test_data.mat')

    X = data['X'].T
    Y = data['y'].T
    nb_examples = X.shape[1]

    mini_batches = random_mini_batches(X, Y, mini_batch_size=64)

    np.testing.assert_equal(mini_batches[0][0].shape, (2, 64))
    np.testing.assert_equal(mini_batches[1][0].shape, (2, 64))
    np.testing.assert_equal(mini_batches[2][0].shape, (2, 64))
    np.testing.assert_equal(mini_batches[3][0].shape, (2, nb_examples - 3*64))
    np.testing.assert_equal(mini_batches[0][1].shape, (1, 64))
    np.testing.assert_equal(mini_batches[1][1].shape, (1, 64))
    np.testing.assert_equal(mini_batches[2][1].shape, (1, 64))
    np.testing.assert_equal(mini_batches[3][1].shape, (1, nb_examples - 3*64))


def test_initialization_weights():

    W0, b0 = initialize_weights([200, 20], "random")
    W1, b1 = initialize_weights([200, 20], "he")

    assert (0.0099 < np.std(W0[1].flatten()) < 0.0101)
    assert (0.099 < np.std(W1[1].flatten()) < 0.101)
    np.testing.assert_equal(b0[1], np.zeros((20, 1)))
    np.testing.assert_equal(b1[1], np.zeros((20, 1)))


def test_dropout_forward():

    marging = 0.01

    A = np.ones((200, 20))
    A0, D0 = dropout_forward(A, keep_prob=1.)
    A1, D1 = dropout_forward(A, keep_prob=0.77)

    assert (A0[0, 0] == 1.)
    assert (sum(sum(A0)) == 200*20)
    assert (sum(sum(D0)) == 200*20)
    assert (np.any(A1[0] == 1/0.77))
    assert (np.any(A1[0] == 0))
    assert (200*20*(1-marging) < sum(sum(A1)) < 200*20*(1+marging))
    assert (200*20*(1-marging)*0.77 < sum(sum(D1)) < 200*20*(1+marging)*0.77)


def test_dropout_backward():

    mat = np.ones((200, 20))
    A, D = dropout_forward(mat, keep_prob=0.77)
    A1 = dropout_backward(mat, D, keep_prob=0.77)

    np.testing.assert_equal(A, A1)


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

    # test_initialization_weights()
    # test_dropout_forward()
    # test_dropout_backward()
    # test_minibatch_subfunction()
    # test_xor()
    test_data()
    # test_optimizer()
    # test_regularization()
    # test_dropout()
    # test_minibatch()


