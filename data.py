from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass
class Image:
    """
    Image class.
    """

    def __init__(self, label: int, image: np.ndarray) -> None:
        """
        Constructor for the Image class.

        Parameters
        ----------
        param `label`: int
            The label of the image.
        param `image`: np.ndarray
            The image.
        """

        self.label: int = label  # The label of the image
        self.image: np.ndarray = image  # The image
        self.resulted_label: int | None = None  # The label that the network predicted

    def __str__(self) -> str:
        return f"Label: {self.label}\n{np.array2string(self.image, precision=3, separator=', ')}"


def get_mnist(
    path: str,
    size: int = 1000,
    for_convolution: bool = False,
    image_size: tuple[int, int] = (28, 28),
) -> List[Image]:
    """
    Gets the MNIST dataset.

    Parameters
    ----------
    param `path`: str
        The path to the csv file.
    param `size`: int
        The size of the dataset to get.
        Default: 1000

    Returns
    -------
    return: List[Image]
    """

    # Read the data from the csv file
    try:
        data = np.array(pd.read_csv(path, header=0, nrows=size))
    except FileNotFoundError:
        # Check if the file exists but with the .gz extension
        if path.endswith(".gz"):
            path = path[:-3]
        else:
            path += ".gz"

        try:
            # Extract the file
            import gzip

            with gzip.open(path, "rb") as f:
                file_content = f.read()

                # Write the file
                with open(path[:-3], "wb") as f:
                    f.write(file_content)

            # Read the data from the csv file
            data = np.array(pd.read_csv(path[:-3], header=0, nrows=size))

        except FileNotFoundError:
            raise FileNotFoundError("The file was not found.")

    # Split the data into labels and images
    labels = data[:, 0]
    images = data[:, 1:].astype(np.float32) / 255

    # Create a list of images
    images = [Image(label, image) for label, image in zip(labels, images)]

    # If the images are for convolution, reshape them
    if for_convolution:
        for i in range(len(images)):
            images[i].image = np.reshape(
                images[i].image, (1, image_size[0], image_size[1])
            )

    return images
