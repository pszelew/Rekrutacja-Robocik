from chess.board import Board
from chess.figures.queen import Queen
from chess.common.color import Color
import sys
def run(file: str):
    b = Board(file)
    fig = Queen(Color.WHITE, 0, 0)
    print(b.check(Color.WHITE))
    #q = Queen(Color.BLACK, 0, 0)
    print("Koniec")