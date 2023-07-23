import numpy as np

from neural_objects.activation import Sigmoid


class LearningRateFunction:
    def linear(self, learning_rate: float, epoch: int) -> float:
        return learning_rate

    def custom_1(self, learning_rate: float, epoch: int) -> float:
        return np.exp(-learning_rate * epoch) / 5

    def custom_2(self, learning_rate: float, epoch: int) -> float:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        return 1 - sigmoid(learning_rate * epoch) / 2.5

    def exponential(self, learning_rate: float, epoch: int) -> float:
        return learning_rate * np.exp(-learning_rate * epoch)
