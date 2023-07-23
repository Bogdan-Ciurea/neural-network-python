import numpy as np
from typing import List
from data import Image
from neural_objects.activation import ReLU, Sigmoid, Tanh
from neural_objects.end_layer import Softmax

from neural_objects.default_layer import DefaultLayer
from neural_objects.layer import Convolutional, Layer, Reshape
from neural_objects.learning_rate_functions import LearningRateFunction


class Network:
    """
    Network class.
    """

    def __init__(self, network: List[DefaultLayer] | str) -> None:
        """
        Constructor for the Network class.
        :param network: The network that will be used.
        :type network: List[DefaultLayer] | str

        :return: None
        """

        # If the network is a string, then we will load it from a file
        if isinstance(network, str):
            self.load(network)
        else:  # Else we will just set the network
            self.network: List[DefaultLayer] = network

        # This will be the data that the network will use to train and test
        self.training_data: List[Image] = []
        self.test_data: List[Image] = []

        # With this array we will be able to draw a graph of the accuracy over a training period
        # The array will be formed of float tuple.
        # The first value will be the one from the training and the last one will be the one from the actual tests
        self.test_graph = []

        # This will be the images that the network misread
        self.misread_images: List[Image] = []

        self.learning_rate_function = LearningRateFunction().linear

    def set_input(self, training_data: List[Image], test_data: List[Image]) -> None:
        """
        This function will set the input for the network.

        :param training_data: The data that will be used to train the network.
        :type training_data: List[Image]

        :param test_data: The data that will be used to test the network.
        :type test_data: List[Image]
        """

        self.training_data = training_data
        self.test_data = test_data

    def set_learning_rate_function(self, function: str) -> None:
        """
        This function will set the learning rate function for the network.

        :param function: The learning rate function.
        :type function: function
        """

        if function == "linear":
            self.learning_rate_function = LearningRateFunction().linear
        elif function == "custom_1":
            self.learning_rate_function = LearningRateFunction().custom_1
        elif function == "custom_2":
            self.learning_rate_function = LearningRateFunction().custom_2
        elif function == "exponential":
            self.learning_rate_function = LearningRateFunction().exponential
        else:
            print("The function that you have entered does not exist!")

    def test(self, save_errors: bool = False) -> float:
        """
        This function will test the network and return the accuracy of the network on the test data.
        """

        if save_errors:
            self.misread_images = []

        errors: List[int] = []
        for image in self.test_data:
            # change the shape of the image
            if len(image.image.shape) == 1:
                image.image.shape += (1,)

            result = image.image
            for obj in self.network:
                result = obj.forward(result)

            if np.argmax(result) == image.label:
                errors.append(1)
            else:
                if save_errors:
                    image.resulted_label = np.argmax(result)
                    self.misread_images.append(image)

                errors.append(0)

        # print(f"Accuracy: {round(sum(errors)/len(errors) * 100, 2)}%")
        return sum(errors) / len(errors)

    def train(
        self, epochs: int = 10, learning_rate: float = 0.1, debug: bool = True
    ) -> None:
        """
        This function will train the network.

        :param epochs: The number of epochs that the network will train.
        :type epochs: int

        :param learning_rate: The learning rate of the network.
        :type learning_rate: float

        :param debug: If this is True, then the function will print the accuracy of the network after each epoch.
        :type debug: bool

        :return: None
        """

        if not len(self.training_data) or not len(self.test_data):
            print("You will first need to set some input for the network to use!")
            return

        self.test_graph: List[List[float]] = []

        # actual training
        for e in range(epochs):
            correct_tests_results: int = 0
            incorrect_tests_results: int = 0

            for image in self.training_data:
                # transform the data
                expected: np.ndarray = np.zeros((10, 1)) + 0.01
                expected[image.label] = 0.99

                # change the shape of the image
                if len(image.image.shape) == 1:
                    image.image.shape += (1,)

                # pass the data throw the network
                output: np.ndarray = image.image
                for obj in self.network:
                    output = obj.forward(output)

                # see if the output was correct
                if np.argmax(output) == image.label:
                    correct_tests_results += 1
                else:
                    incorrect_tests_results += 1

                # get the error
                error: np.ndarray = output - expected

                # use the error for back propagation
                for obj in reversed(self.network):
                    error = obj.backward(
                        error, self.learning_rate_function(learning_rate, e)
                    )

            if e + 1 == epochs:
                tests_tuple = [
                    correct_tests_results
                    / (correct_tests_results + incorrect_tests_results),
                    self.test(True),
                ]
            else:
                tests_tuple = [
                    correct_tests_results
                    / (correct_tests_results + incorrect_tests_results),
                    self.test(),
                ]

            if debug:
                print("----------------")
                print(f"epoch {e + 1}")
                print("----------------")
                print(f"Accuracy on training: {round(tests_tuple[0] * 100, 2)}%")
                print(f"Accuracy on tests:    {round(tests_tuple[1] * 100, 2)}%")

            self.test_graph.append(tests_tuple)

    def draw_graph(self) -> None:
        """
        This function will draw a graph of the accuracy over a training period.
        It is required that the network was trained before calling this function.

        :return: None
        """

        if not len(self.test_graph):
            print("First you will have to train the network!")
            return

        import matplotlib.pyplot as plt

        # plotting the points
        temp_tests_results = np.transpose(self.test_graph)
        plt.plot(temp_tests_results[0], label="Training Accuracy")
        plt.plot(temp_tests_results[1], label="Test Accuracy")

        # naming the x axis
        plt.xlabel("Epochs")
        # naming the y axis
        plt.ylabel("Accuracy")

        # giving a title to my graph
        plt.title("Accuracy graph")

        # function to show the plot
        plt.show()

    def draw_errors(self, number: int = 10) -> None:
        """
        This function will draw the images that the network misread.
        It is required that the network was trained before calling this function.

        :param number: The number of images that will be drawn.
        :type number: int

        :return: None
        """

        if not len(self.test_graph):
            print("First you will have to train the network!")
            return

        import matplotlib.pyplot as plt

        i: int = 1

        for image in self.misread_images:
            image.image *= 255
            image.image = np.array(image.image, dtype="int64")
            if len(image.image.shape) != 2:
                image.image = image.image.reshape((28, 28))
            plt.title(f"Expected: {image.label} Got: {image.resulted_label}")
            plt.imshow(image.image, cmap="gray")
            plt.show()

            if i == number:
                break
            else:
                i += 1

    def save(self, path: str) -> None:
        """
        This function will save the network to a file.

        :param path: The path to the file.
        :type path: str

        :return: None
        """

        # Open the file
        file = open(path, "w")

        # Write the number of layers
        file.write(f"{len(self.network)}\n")

        # Write the name of the layers
        for obj in self.network:
            file.write(f"{obj.__class__.__name__}\n")

        # Close the file and let the layers write their own data
        file.close()

        for obj in self.network:
            # Save the information only if the layer is a Layer
            if (
                obj.__class__.__name__ == "Layer"
                or obj.__class__.__name__ == "Convolutional"
                or obj.__class__.__name__ == "Reshape"
            ):
                obj.save(path)

    def load(self, path: str) -> None:
        """
        This function will load the network from a file.

        :param path: The path to the file.
        :type path: str

        :return: None
        """

        # Clear the network
        self.network = []

        # Open the file
        file = open(path, "r")

        # Read the number of layers
        number_of_layers = int(file.readline())

        layer_names: List[str] = []
        # Read the name of the layers and store them
        for i in range(number_of_layers):
            layer_name = file.readline().replace("\n", "")
            layer_names.append(layer_name)

        # Iterate throw the layers and create them
        for i in range(number_of_layers):
            # Get the name of the layer
            layer_name = layer_names[i]

            # Create the layer
            if layer_name == "Layer":
                # Read the size of the layer
                input_size: int = int(file.readline())
                output_size: int = int(file.readline())

                # Read the weights
                weights: np.ndarray = np.zeros((output_size, input_size))
                for i in range(output_size):
                    weights[i] = np.array(file.readline().replace("\n", "").split(" "))[
                        :-1
                    ]

                # Read the biases
                biases: np.ndarray = np.array(
                    file.readline().replace("\n", "").split(" ")
                )[:-1]

                # Convert the biases to a 2D array
                biases = biases.reshape((output_size, 1))

                # Convert the data to float
                weights = weights.astype(np.float)
                biases = biases.astype(np.float)

                # Add the layer to the network
                self.network.append(Layer(weights, biases))

            elif layer_name == "Convolutional":
                # The first line will be of the format "(input_depth, input_height, input_width)"
                input_shape = tuple(
                    map(
                        int,
                        file.readline()
                        .replace("\n", "")
                        .replace("(", "")
                        .replace(")", "")
                        .split(", "),
                    )
                )

                # The second line will be of the format "depth"
                depth = int(file.readline())

                # The third line will be of the format "kernel_size"
                kernel_size = int(file.readline())
                kernel_shape = (depth, input_shape[0], kernel_size, kernel_size)

                # The next depth * kernel_size lines will be the weights
                weights = np.zeros((depth, input_shape[0], kernel_size, kernel_size))
                for i in range(depth):
                    for j in range(kernel_size):
                        weights[i][0][j] = np.array(
                            file.readline().replace("\n", "").split(" ")
                        )[:-1]

                # Build the output shape
                output_shape = (
                    depth,
                    input_shape[1] - kernel_size + 1,
                    input_shape[2] - kernel_size + 1,
                )

                # The next depth lines will be the biases
                # The will be of the format depth * output_height * output_width
                biases = np.zeros((depth, output_shape[1], output_shape[2]))
                for i in range(depth):
                    for j in range(output_shape[1]):
                        biases[i][j] = np.array(
                            file.readline().replace("\n", "").split(" ")
                        )[:-1]

                # Convert the data to float
                weights = weights.astype(np.float)
                biases = biases.astype(np.float)

                # Add the layer to the network
                layer = Convolutional(input_shape, depth, kernel_size)
                layer.load(weights, biases)

                self.network.append(layer)

            elif layer_name == "Reshape":
                # Read the input shape
                input_shape = tuple(
                    map(
                        int,
                        file.readline()
                        .replace("\n", "")
                        .replace("(", "")
                        .replace(")", "")
                        .split(", "),
                    )
                )

                # Read the output shape
                output_shape = tuple(
                    map(
                        int,
                        file.readline()
                        .replace("\n", "")
                        .replace("(", "")
                        .replace(")", "")
                        .split(", "),
                    )
                )

                # Add the layer to the network
                self.network.append(Reshape(input_shape, output_shape))

            elif layer_name == "ReLU":
                self.network.append(ReLU())

            elif layer_name == "Sigmoid":
                self.network.append(Sigmoid())

            elif layer_name == "Tanh":
                self.network.append(Tanh())

            elif layer_name == "Softmax":
                self.network.append(Softmax())

        # Close the file
        file.close()
