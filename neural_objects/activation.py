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


class Linear(Activation):
    """
    Linear activation class.
    """

    def __init__(self) -> None:
        def linear(x):
            return x

        def linear_prime(x):
            return 1

        super().__init__(linear, linear_prime)


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


class LeakyReLU(Activation):
    """
    LeakyReLU activation class.
    """

    def __init__(self) -> None:
        def leaky_relu(x):
            return np.maximum(0.01 * x, x)

        def leaky_relu_prime(x):
            return 0.01 * (x > 0) + 1 * (x <= 0)

        super().__init__(leaky_relu, leaky_relu_prime)


class ELU(Activation):
    """
    ELU activation class.
    """

    def __init__(self) -> None:
        def elu(x):
            return np.maximum(0.01 * (np.exp(x) - 1), x)

        def elu_prime(x):
            return 0.01 * np.exp(x) * (x > 0) + 1 * (x <= 0)

        super().__init__(elu, elu_prime)


class SELU(Activation):
    """
    SELU activation class.
    """

    def __init__(self) -> None:
        def selu(x):
            return 1.0507 * np.maximum(0.01 * (np.exp(x) - 1), x)

        def selu_prime(x):
            return 1.0507 * (0.01 * np.exp(x) * (x > 0) + 1 * (x <= 0))

        super().__init__(selu, selu_prime)


class BinaryStep(Activation):
    """
    BinaryStep activation class.
    """

    def __init__(self) -> None:
        def binary_step(x):
            return np.heaviside(x, 1)

        def binary_step_prime(x):
            return 0

        super().__init__(binary_step, binary_step_prime)


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


class Swish(Activation):
    """
    Swish activation class.
    """

    def __init__(self) -> None:
        def swish(x):
            return x * Sigmoid().forward(x)

        def swish_prime(x):
            return x * Sigmoid().forward(x) + Sigmoid().backward(x, 1)

        super().__init__(swish, swish_prime)
