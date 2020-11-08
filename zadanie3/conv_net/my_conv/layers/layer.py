from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Layer():
    def __init__(self, name: str, input_dims: tuple, output_dims: tuple):
        self.name = name
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input = np.zeros((input_dims))
        self.activation_val = np.zeros((output_dims))
        self.output = np.zeros((output_dims))