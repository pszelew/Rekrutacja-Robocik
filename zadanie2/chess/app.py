from chess.board import Board
from chess.figures.queen import Queen
from chess.common.color import Color
import sys
def run(file: str):
    b = Board(file)
    # Create board
    print("Wczytany widok planszy:")
    print(b)
    win_black = b.check_mate(Color.BLACK)
    # Check if black can win
    win_white = b.check_mate(Color.WHITE)
    # Check if white can win
    if win_black[0] is not None:
        if win_black[0] is -1:
            print(f"Czarny juz wygral jesli to jego ruch!!!")
        else:    
            s_d = win_black[0]
            #start down
            s_r = win_black[1]
            #start right
            e_d = win_black[2]
            #end down
            e_r = win_black[3]
            #end right
            print("Czarny może wygrać ({}{}-{}{})".format(b.dic_ver[s_d], b.dic_hor[s_r], b.dic_ver[e_d], b.dic_hor[e_r]))
        return
    if win_white[0] is not None:
        if win_white[0] is -1:
            print(f"Bialy juz wygral jesli to jego ruch!!!")
        else:    
            s_d = win_white[0]
            #start down
            s_r = win_white[1]
            #start right
            e_d = win_white[2]
            #end down
            e_r = win_white[3]
            #end right
            print("Bialy może wygrać ({}{}-{}{})".format(b.dic_ver[s_d], b.dic_hor[s_r], b.dic_ver[e_d], b.dic_hor[e_r]))
        return
    print("Niestety zadna ze stron nie moze osiagnac teraz zwyciestwa")