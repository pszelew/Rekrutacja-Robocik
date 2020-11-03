from chess.board import Board
from chess.figures.queen import Queen
from chess.common.color import Color
import sys
def run(file: str):
    b = Board(file)
    # Create board
    print("Wczytany widok planszy:")
    print(b)
    temp_str: str
    win_black: list
    win_white: list
    win_black = b.check_mate(Color.BLACK)
    # Check if black can win
    win_white = b.check_mate(Color.WHITE)
    # Check if white can win
    if win_black:
        if win_black[0][0] is -1:
            print(f"Czarny juz wygral jesli to jego ruch!!!")
        else:    
            temp_str = "Czarny moze wygrac: " 
            for i, move in enumerate(win_black):
                s_d = move[0]
                #start down
                s_r = move[1]
                #start right
                e_d = move[2]
                #end down
                e_r = move[3]
                #end right
                temp_str += "({}{}-{}{})".format(b.dic_hor[s_r], b.dic_ver[s_d], b.dic_hor[e_r], b.dic_ver[e_d])
                if i < (len(win_black.count) - 1):
                    temp_str += ", "
            print(temp_str)
        return
    if win_white:
        if win_white[0][0] is -1:
            print(f"Bialy juz wygral jesli to jego ruch!!!")
        else:    
            temp_str = "Bialy moze wygrac: " 
            for i, move in enumerate(win_white):
                s_d = move[0]
                #start down
                s_r = move[1]
                #start right
                e_d = move[2]
                #end down
                e_r = move[3]
                #end right
                temp_str += "({}{}-{}{})".format(b.dic_hor[s_r], b.dic_ver[s_d], b.dic_hor[e_r], b.dic_ver[e_d])
                if i < (len(win_white) - 1):
                    temp_str += ", "
            print(temp_str)
        return
    print("Niestety zadna ze stron nie moze osiagnac teraz zwyciestwa")