import numpy as np
from neural_objects.default_layer import DefaultLayer


class Activation(DefaultLayer):
    """
    Activation class.
    """

    def __init__(self, activation, activation_prime) -> None:
        """
        Constructor for the Activation class.

        Parameters
        ----------
        param `activation`: function
            The activation function.
        param `activation_prime`: function
            The derivative of the activation function.

        Returns
        -------
        return: None
        """

        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass for the softmax layer.

        Parameters
        ----------
        param `input`: np.ndarray
            The input to the layer.
        """

        self.input: np.ndarray = input
        return self.activation(self.input)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for the softmax layer.

        Parameters
        ----------
        param `output_gradient`: np.ndarray
            The gradient of the output.
        param `learning_rate`: float
            The learning rate of the network. NOT USED.

        Returns
        -------
        return: np.ndarray
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Tanh(Activation):
    """
    Tanh activation class.
    """

    def __init__(self) -> None:
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    """
    ReLU activation class.
    """

    def __init__(self) -> None:
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return x > 0

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    """
    Sigmoid activation class.
    """

    def __init__(self) -> None:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
