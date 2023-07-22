from data import *
from network import Network
from neural_objects.activation import ReLU, Sigmoid, Tanh
from neural_objects.end_layer import Softmax
from neural_objects.default_layer import DefaultLayer
from neural_objects.layer import Convolutional, Layer, Reshape

if __name__ == "__main__":
    print("Loading data...")

    # Get the data
    training_data = get_mnist("data/mnist_train.csv", 60000, for_convolution=True)
    test_data = get_mnist("data/mnist_test.csv", 10000, for_convolution=True)

    print("Data loaded!")

    # Create the network
    network: List[DefaultLayer] = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((3, 24, 24), (3 * 24 * 24, 1)),
        Layer(3 * 24 * 24, 40),
        ReLU(),
        Layer(40, 10),
        Tanh(),
        Softmax(),
    ]

    # Create the network
    neural_network = Network(network)
    # neural_network = Network("network.txt")  # Load the network from a file
    neural_network.set_input(training_data, test_data)  # Set the input of the network

    print("Training...")  # Train the network
    neural_network.train(epochs=10, learning_rate=0.1, debug=True)
    print("Training done!")

    # neural_network.draw_graph()

    # neural_network.draw_errors()

    # Save the network
    # neural_network.save("network.txt")
