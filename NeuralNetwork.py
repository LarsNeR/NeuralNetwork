import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class NeuralNetwork(object):

    def __init__(self, n_hidden_nodes, n_hidden_layers=1, epsilon=1):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers + 2
        self.layers = list(np.zeros(self.n_layers))
        self.weights = list(np.zeros(self.n_layers - 1))
        self.epsilon = epsilon

    def fit(self, X, y):
        self.random_initialization(X)
        self.lbfgs(X, y)

    def random_initialization(self, X):
        """
        Random initialize the weights. Necessary for neural networks
        """
        self.weights[0] = np.random.rand(self.n_hidden_nodes, X.shape[1] + 1) * (2 * self.epsilon) - self.epsilon
        for layer in range(1, self.n_hidden_layers):
            self.weights[layer] = np.random.rand(self.n_hidden_nodes, self.n_hidden_nodes + 1) * (2 * self.epsilon) - self.epsilon
        self.weights[self.n_hidden_layers] = np.random.rand(1, self.n_hidden_nodes + 1) * (2 * self.epsilon) - self.epsilon

    def lbfgs(self, X, y):
        weights_vec = self.unroll_weights()
        min, ymin, dic = fmin_l_bfgs_b(func=self.cost, x0=weights_vec, args=(X, y), approx_grad=True, maxiter=1000)

    def unroll_weights(self):
        """
        Unroll shaped weights. Necessary for lbfgs
        """
        weights_vec = []
        for layer in range(len(self.weights)):
            weights_vec = np.append(weights_vec, np.ravel(self.weights[layer]))
        return weights_vec

    def cost(self, *args):
        """
        Cost-function which is passed to lbfgs for minimizing it
        """
        weights_vec = args[0]
        X = args[1]
        y = args[2]
        predictions = self.forward_propagation(X, weights_vec)
        return - sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) / X.shape[1]

    def forward_propagation(self, X, weights_vec):
        self.reshape_weights(X, weights_vec)

        predictions = np.empty(0)

        self.layers[0] = np.c_[np.ones(X.shape[0]), X]
        for layer in range(1, self.n_hidden_layers + 1):
            self.layers[layer] = self.hypothesis(self.layers[layer - 1], self.weights[layer - 1])
            self.layers[layer] = np.c_[np.ones(self.layers[layer].shape[0]), self.layers[layer]]
        self.layers[self.n_layers - 1] = self.hypothesis(self.layers[self.n_layers - 2], self.weights[self.n_hidden_layers])
        predictions = np.ravel(self.layers[self.n_layers-1])
        return predictions

    def reshape_weights(self, X, weights_vec):
        self.weights[0] = weights_vec[0 : self.weights[0].size].reshape(self.n_hidden_nodes, X.shape[1] + 1)
        for layer in range(1, self.n_hidden_layers):
            self.weights[layer] = weights_vec[self.weights[layer-1].size : self.weights[self.n_hidden_layers-1].size+self.weights[layer].size].reshape(self.n_hidden_nodes, self.n_hidden_nodes + 1)
        self.weights[self.n_hidden_layers] = weights_vec[self.weights[self.n_hidden_layers-1].size : self.weights[self.n_hidden_layers-1].size+self.weights[self.n_hidden_layers].size].reshape(1, self.n_hidden_nodes + 1)


    def hypothesis(self, X, weights):
        return 1 / (1 + np.exp(-np.dot(X, weights.T)))

    def predict(self, X):
        """
        Predict the target class(es)

        Parameters
        ----------
        X : numpy array, shape(m_samples, n_features)
            Data to use for prediction

        Returns
        -------
        prediction : numpy array, shape(m_samples, )
            Predicted class for every sample in X
        """
        weights_vec = self.unroll_weights()
        return np.where(self.forward_propagation(X, weights_vec) >= 0.5, 1, 0)
