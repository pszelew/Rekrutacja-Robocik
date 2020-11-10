from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Layer():
    """
    A class used to represent layer of neural network

    Attributes
    ----------
    name : str
        Full name of layer
    input_dims: tuple
        Input dimensions of layer
    output_dims: tuple
        Color of the figure
    down: int
        Current vertical position on board
    right: int
        Current horizontal position on board
    Methods
    -------
    check_moves() -> list[tuple[int, int]]
        Returns list of possible moves
    move(pos: tuple[int, int])
        Move to given possition
    """
    def __init__(self, name: str, input_dims: tuple, output_dims: tuple):
        self.name = name
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input = np.zeros((input_dims))
        self.activation_val = np.zeros((output_dims))
        self.output = np.zeros((output_dims))

    