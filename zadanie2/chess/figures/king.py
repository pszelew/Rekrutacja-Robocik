from __future__ import annotations
from chess.figures.figure import Figure
from chess.common.color import Color
from chess.common.possible_move import PossibleMove
class King(Figure):
    """
    A  class used to represent a King figure from the game of chess

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
        self.name = "king"
        self.short = "W"
        self.pos_moves.append(PossibleMove(1, 1, False))
        self.pos_moves.append(PossibleMove(1, -1, False))
        self.pos_moves.append(PossibleMove(-1, 1, False))
        self.pos_moves.append(PossibleMove(-1, -1, False))
        self.pos_moves.append(PossibleMove(1, 0, False))
        self.pos_moves.append(PossibleMove(-1, 0, False))
        self.pos_moves.append(PossibleMove(0, 1, False))
        self.pos_moves.append(PossibleMove(0, -1, False))