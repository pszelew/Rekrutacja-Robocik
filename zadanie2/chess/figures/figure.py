from __future__ import annotations
from abc import ABC, abstractmethod
from chess.common.color import Color
from chess.common.possible_move import PossibleMove
class Figure(ABC):
    """
    An abstract class used to represent a figure from the game of chess

    Attributes
    ----------
    name : str
        Full name of figure
    short: str
        Short name of figure
    color: Color
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

    def __init__(self, color: Color, down: int, right: int):
        """
        Parameters
        ----------
        color: Color
            Color of the figure
        down: int
            Vertical position on the board
        right: int
            Horizontal position on the board
        """
        self.name: str
        self.short: str
        self.color = color
        self.down = down
        self.right = right
        self.pos_moves: list[PossibleMove]
        self.pos_moves = []
    def __repr__(self):
        if self.color == Color.BLACK:
            return "black_" + self.name
        else:
            return "white_" + self.name
    def __str__(self):
        if self.color == Color.BLACK:
            return "black_" + self.name
        else:
            return "white_" + self.name