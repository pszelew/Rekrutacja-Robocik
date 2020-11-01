from __future__ import annotations
from chess.figures.figure import Figure
from chess.common.color import Color
from chess.common.possible_move import PossibleMove
class Pawn(Figure):
    """
    A  class used to represent a Pawn figure from the game of chess

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
        color : Color
            Color of the figure
        down: int
            Vertical position on the board
        right: int
            Horizontal position on the board
        """
        super().__init__(color, down, right)
        self.name = "pawn"
        self.short = "p"
        if self.color == Color.BLACK:
            self.pos_moves.append(PossibleMove(1, 0, False))
        else:
            self.pos_moves.append(PossibleMove(-1, 0, False))