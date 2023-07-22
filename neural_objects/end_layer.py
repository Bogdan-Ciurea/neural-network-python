import numpy as np
from neural_objects.default_layer import DefaultLayer


class Softmax(DefaultLayer):
    """
    Softmax layer class.
    """

    def __init__(self) -> None:
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass for the softmax layer.

        Parameters
        ----------
        param `input`: np.ndarray
            The input to the layer.
        """
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

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

        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
