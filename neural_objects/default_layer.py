import numpy as np


class DefaultLayer:
    """
    Default layer class.
    It's a base class for all layers.
    """

    def __init__(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self) -> None:
        pass

    def forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass
