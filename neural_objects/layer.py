from scipy import signal
import numpy as np
from neural_objects.default_layer import DefaultLayer


class Layer(DefaultLayer):
    """
    Layer class.
    """

    def __init__(
        self, input_size: int | np.ndarray, output_size: int | np.ndarray
    ) -> None:
        """
        Constructor for the Layer class.

        Parameters
        ----------
        param `input_size`: int | np.ndarray
            The input size of the layer.
            If np.ndarray, then the weights and biases are set to the input_size and output_size respectively.
            If int, then the weights and biases are set to random values between -1 and 1.
        param `output_size`: int | np.ndarray
            The output size of the layer.
            If np.ndarray, then the weights and biases are set to the input_size and output_size respectively.
            If int, then the weights and biases are set to random values between -1 and 1.

        Returns
        -------
        return: None

        """
        if isinstance(input_size, np.ndarray):
            self.input_size: int = input_size.shape[1]
            self.output_size: int = output_size.shape[0]
            self.weights: np.ndarray = input_size
            self.biases: np.ndarray = output_size
        else:
            self.input_size: int = input_size
            self.output_size: int = output_size
            self.weights: np.ndarray = np.random.uniform(
                -1, 1, (output_size, input_size)
            )
            self.biases: np.ndarray = np.random.uniform(-1, 1, (output_size, 1))

    def save(self, path: str) -> None:
        """
        Saves the layer to a file. The file is appended to, so if you want to save multiple layers to the same file,
        make sure to delete the file before saving.

        Parameters
        ----------
        param `path`: str
            The path to the file to save the layer to.
        """

        # Open the file, and append the weights and biases
        file = open(path, "a")
        file.write(f"{self.input_size}\n")
        file.write(f"{self.output_size}\n")

        for i in range(self.output_size):
            for j in range(self.input_size):
                file.write(f"{self.weights[i][j]} ")
            file.write("\n")

        for i in range(self.output_size):
            file.write(f"{self.biases[i][0]} ")
        file.write("\n")

        file.close()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.

        Parameters
        ----------
        param `inputs`: np.ndarray

        Returns
        -------
        return: np.ndarray
        """

        self.inputs: np.ndarray = inputs
        return np.dot(self.weights, self.inputs) + self.biases

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass of the layer.

        Parameters
        ----------
        param `output_gradient`: np.ndarray
        param `learning_rate`: float

        Returns
        -------
        return: np.ndarray
        """

        weights_gradient = np.dot(output_gradient, self.inputs.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class Convolutional(DefaultLayer):
    def __init__(self, input_shape: tuple, depth: int, kernel_size: int) -> None:
        """
        Constructor for the Convolutional class.

        Parameters
        ----------
        param `input_shape`: tuple
            The shape of the input. Consists of the depth, height and width.

        param `depth`: int
            The depth of the output.

        param `kernel_size`: int
            The size of the kernel.

        """

        input_depth, input_height, input_width = input_shape
        self.input_shape: tuple[int] = input_shape

        self.depth: int = depth
        self.input_shape: int = input_shape
        self.input_depth: int = input_depth
        self.output_shape: tuple[int] = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernel_size: int = kernel_size
        self.kernels_shape: tuple[int] = (depth, input_depth, kernel_size, kernel_size)
        self.kernels: np.ndarray = np.random.randn(*self.kernels_shape)
        self.biases: np.ndarray = np.random.randn(*self.output_shape)

    def save(self, path: str) -> None:
        """
        Saves the layer to a file. The file is appended to, so if you want to save multiple layers to the same file,
        make sure to delete the file before saving.

        Parameters
        ----------
        param `path`: str
            The path to the file to save the layer to.
        """

        # Open the file, and append the weights and biases
        file = open(path, "a")
        file.write(f"{self.input_shape}\n")
        file.write(f"{self.depth}\n")
        file.write(f"{self.kernel_size}\n")

        # Write the kernels
        for i in range(self.depth):
            for j in range(self.input_depth):
                for k in range(self.kernels_shape[2]):
                    for l in range(self.kernels_shape[3]):
                        file.write(f"{self.kernels[i][j][k][l]} ")
                    file.write("\n")

        # Write the biases
        for i in range(self.depth):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    file.write(f"{self.biases[i][j][k]} ")
                file.write("\n")

        file.close()

    def load(self, kernels: np.ndarray, biases: np.ndarray) -> None:
        """
        Loads the layer from a file.

        Parameters
        ----------
        param `kernels`: np.ndarray
            The kernels to load.
        param `biases`: np.ndarray
            The biases to load.
        """

        self.kernels = kernels
        self.biases = biases

    def forward(self, input) -> np.ndarray:
        """
        Forward pass of the layer.

        Parameters
        ----------
        param `inputs`: np.ndarray

        Returns
        -------
        return: np.ndarray
        """

        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )

        return self.output

    def backward(self, output_gradient, learning_rate) -> np.ndarray:
        """
        Backward pass of the layer.

        Parameters
        ----------
        param `output_gradient`: np.ndarray
        param `learning_rate`: float

        Returns
        -------
        return: np.ndarray
        """

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class Reshape(DefaultLayer):
    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        """
        Constructor for the Reshape class.

        Parameters
        ----------
        param `input_shape`: tuple
            The shape of the input.

        param `output_shape`: tuple
            The shape of the output.

        """

        self.input_shape: tuple = input_shape
        self.output_shape: tuple = output_shape

    def save(self, path: str) -> None:
        """
        Saves the layer to a file. The file is appended to, so if you want to save multiple layers to the same file,
        make sure to delete the file before saving.

        Parameters
        ----------
        param `path`: str
            The path to the file to save the layer to.
        """

        # Open the file, and append the weights and biases
        file = open(path, "a")
        file.write(f"{self.input_shape}\n")
        file.write(f"{self.output_shape}\n")
        file.close()

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.

        Parameters
        ----------
        param `inputs`: np.ndarray

        Returns
        -------
        return: np.ndarray
        """

        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass of the layer.

        Parameters
        ----------
        param `output_gradient`: np.ndarray
        param `learning_rate`: float

        Returns
        -------
        return: np.ndarray
        """

        return np.reshape(output_gradient, self.input_shape)
