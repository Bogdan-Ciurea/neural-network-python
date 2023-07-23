# Neural Network in Python

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Model](#model)
6. [Results](#results)

## Introduction

This project is a simple neural network written in Python using only the NumPy library. It is designed to be easy to use and extend. This project started as preparations for a university project (that never took place) and is now a project to learn more about neural networks and Python.

## Installation

0. (optional) If you are using pyenv, use the version of Python specified in _.python-version_ by running _pyenv local_ in the project directory.
1. Install the required packages using `pip install -r requirements.txt`.
2. Run `python main.py` to train the model and evaluate it on the test set.

## Usage

The project is designed to be easy to use and extend.
In order to train the model, you can run `python main.py` in the project directory.

The default neural network is a simple CNN with 3 convolutional layers and 2 fully connected layers that is trained on the MNIST dataset.

### Layers

You can change the layers of the neural network by changing the `layers` variable in `main.py`.

The layers are defined in the **neural_objects** folder.

The available types of layers are:

- Convolutional Layer
- Dense Layer (the name in the project is just _Layer_)
- Flatten Layer

Furthermore you can change the activation function of the layers by just adding them after a layer in the `layers` variable.

The available activation functions are:

- Linear
- Tanh
- ReLU
- Leaky ReLU
- ELU
- SELU
- Binary Step
- Sigmoid
- Swish
- Softmax

### Learning Rate

The learning rate can be changed by changing of the neural network is going to be changed in two places. The first place is in the `main.py` file, in the `learning_rate` variable. The second place is in the `neural_network/learning_rate_functions.py` file, that contain the learning rate functions. The learning rate functions are imported in the `main.py` file and are used in the `NeuralNetwork` class.

The two custom learning rate functions are:

```
np.exp(-learning_rate * epoch) / 5  and
1 - sigmoid(learning_rate * epoch) / 2.5
```

These 3 functions (the custom + the exponential) can be seen as:
![Learning Functions](results/learning_functions.png)

## Data

You can use different datasets by changing the path variables in `main.py`.

Keep in mind that the you can specify if the read images should be in an array or a matrix in the `read_data` function in `main.py`. You can also specify the image size there.

## Model

The architecture of this project is going to be simple. The neural network's objects are going to be defined in the **neural_objects** folder. The neural network itself is going to be defined in the **neural_network** file. Furthermore, the readding of the data is going to be found in the **data.py** file.

The **main.py** file is going to be the file that is going to be executed. It is going to call the most important functions in the neural network and data files and use them appropriately.

### Neural Network's objects

The neural network's objects are going to be defined in the **neural_objects** folder, each category having it's own file. The objects are going to be defined as classes.

Because the definition of layers wanted to be simple, the best approach was to use an array that contained them. This way, the layers can be easily added to the neural network.

The layers itself are going to derive from a base class called **DefaultLayer**. This class is going to have the most important functions that are going to be used by the neural network like **forward** and **backward**.

The other files inside the **neural_objects** folder are going to take care of:

1. The activation functions. Found in the **activation.py** file, they contain the activation functions and their derivatives. The activation functions are going to be defined as classes and are going to derive from a base class called **Activation**. This file contains the following activation functions:

   - Tanh
   - ReLU
   - Sigmoid
   - Softmax

2. The end layers. Found in the **end_layer.py** file, they contain the layers that are going to be used at the end of the neural network. The end layers are going to be defined as classes and are going to derive from a base class called **EndLayer**. This file contains the following end layers:

   - Softmax

3. The actual layers. Found in the **layer.py** file, they contain the layers that are going to be used in the neural network. The layers are going to be defined as classes and are going to derive from a base class called **Layer**. This file contains the following layers:

   - Convolutional Layer
   - Dense Layer (the name in the project is just _Layer_)
   - Flatten Layer

   Furthermore, the classes in this file are going to contain two more functions (besides the ones from the **DefaultLayer** class), these being **save** and **load** (found just in the convolutional layer). These functions are going to be used to save and load the weights of the layers from a file.

### Neural Network

The neural network is going to be defined in the **network.py** file. It is going to be defined as a class called **NeuralNetwork**. It's purpose is to manage the relationships between the layers and to train the neural network.

The neural network is going to have the following functions:

- **set_input** - to load the data from the files
- **train** - to train the neural network
- **draw_graph** - to draw the graph of the neural network that will show the accuracy of the neural network on the training and test set
- **draw_errors** - to draw some of the number that the neural network got wrong
- **save** - to save the weights of the neural network to a file
- **load** - to load the weights of the neural network from a file

## Results

From my testing and experimentation, the results did not turn out to be that good. The neural network was able to get an accuracy of 80% on the tests that I did. This is not that good, but it is not that bad either. The neural network was trained for 10 epochs. The most important factor that I did not add as a feature to the network is the fact that, as the number of epochs increases, the learning rate should decrease. This is a feature that I am going to add in the future.

The neural network was trained on the MNIST dataset. The dataset contains 60000 images for training and 10000 images for testing therefore extensive testing cound not be done in a reasonable amount of time.

This is a graph of the accuracy of the neural network on the training and test set:

![Graph](results/graph.png)

### Errors

Even though the neural network got an accuracy of 80%, it still got some of the numbers wrong. Here are some of the numbers that the neural network got wrong:

![Error 1](results/error1.png)
![Error 2](results/error2.png)
![Error 3](results/error3.png)
![Error 4](results/error4.png)

Although the first image is for sure an error (the number is clearly a 2), the other errors can be understood. The second image is a 3, but it is written in a way that it can be understood as a 5 because of it's line in the middle. The third image is a 2, but it is but it's tail is a bit too short. The fourth image is a 9, but it's head is a bit too round.
