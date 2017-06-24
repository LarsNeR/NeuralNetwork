# NeuralNetwork
An example of how to create a simple self-developed Neural Network from scratch in Python

*I'm not sure if I have implemented everything correctly. Please tell me (or better: make a PR) if you have found a mistake.*

#### Main.py
Shows how to use **NeuralNetwork.py**

#### NeuralNetwork.py
Contains the Neural Network using L-BFGS for minimizing the costs.

## How to use
You can pull the repository the way it is. Running **Main.py** calls the Neural Network with the breast_cancer dataset. If you just want to use the Neural Network you have to initialize it with the number of hidden nodes (and optionally: number of hidden layers and epsilon for random initialization) and then call `fit()` with a Numpy Array `X` (m_samples, n_features) and a Numpy Array `y` (m_samples).

## Contribution
If you have an idea how to improve this Neural Network keeping it as simple as possible please fork it and make a PR. Like said before: I'm not 100% sure if it is working correctly.
