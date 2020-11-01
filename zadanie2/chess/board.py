from __future__ import annotations
from chess.figures import bishop
from chess.figures import figure
from chess.figures import king
from chess.figures import knight
from chess.figures import pawn
from chess.figures import queen
from chess.figures import rook
from chess.common.color import Color 
from chess.common.possible_move import PossibleMove
class Board:
    """
    A class used to represent a board from the game of chess

    Attributes
    ----------
    s: list[figures.figure.Figure]
        List keeping state of game

    Methods
    -------
    free(tuple[int, int])
        Sprawdza czy pole jest wolne
    """
    def __init__(self, file: str):
        """
        Parameters
        ----------
        file : str
            The name of the file containing board state to be loaded
        """
        with open(file) as f:
            raw_data = f.read()
        raw_data = raw_data.replace("[", "").replace("]", ",")
        raw_data = raw_data.replace("'", "").replace("\n", "")
        lst_data = raw_data.split(",")
        for index, item in enumerate(lst_data):
            lst_data[index] = item.strip() 
        lst_data = lst_data[:-1]
        self.s = [lst_data[i*8:i*8+8] for i in range(8)]
        temp_color: Color
        temp_class: figure.Figure
        for i, item_i in enumerate(self.s):
            for j, item in enumerate(item_i):
                if item == "--":                                            #empty space
                    self.s[i][j] = None
                    continue
                if item[0] == "b":
                    temp_color = Color.BLACK
                if item[0] == "w":
                    temp_color = Color.WHITE
                if item[1] == "p":                                            #pawn
                    temp_class = pawn.Pawn
                elif item[1] == "r":                                          #rook
                    temp_class = rook.Rook
                elif item[1] == "k":                                          #knight
                    temp_class = knight.Knight
                elif item[1] == "b":                                          #bishop
                    temp_class = bishop.Bishop
                elif item[1] == "q":                                          #queen
                    temp_class = queen.Queen
                elif item[1] == "W":                                          #king
                    temp_class = king.King
                self.s[i][j] = temp_class(temp_color, i, j)                   #create objects
    def free(self, pos: tuple[int, int]) -> bool:
        """Checks if position is free
        Parameters
        ----------
        pos: tuple[int, int]
            Position to check
        
        Returns
        ----------
        bool
            True -- position is free
            False -- position is not free
        """
        if self.s[pos[0]][pos[1]] is None:
           return True
        else:
            return False
    
    def can_move(self, pos: tuple[int, int], fig: figure.Figure) -> bool:
        """Checks if figure can move to position
        Parameters
        ----------
        pos: tuple[int, int]
            Position to check
        fig: figure.Figure
            Figure to be checked
        Returns
        ----------
        bool
            True -- figure can move there
            False -- figure can not move there
        """
        
        if self.free(pos) is False and self.s[pos[0]][pos[1]].color is fig.color: #if occupied by own figure
            return False
    
        to_move: tuple[int, int]
        to_move = (pos[0]-fig.down, pos[1]-fig.right)
        
        if to_move == (0, 0):
            return False
        for mov in fig.pos_moves:
            if mov.down == to_move[0] and mov.right == to_move[1]:          #it is a perfect move
                return True
            if mov.it == False:                                             #it failed
                continue
            f: int
            
            if mov.down is not 0:                                                      #how many iterations to reach target
                f = to_move[0]//mov.down
            else:                                                                               
                f = to_move[1]//mov.right
            if f > 0 and f*mov.down == to_move[0] and f*mov.right == to_move[1]:       #can move there through iterations
                for i in range(f-1):
                    tmp_down: int
                    tmp_right: int
                    tmp_down = fig.down+(i+1)*mov.down
                    tmp_right = fig.right+(i+1)*mov.right
                    if tmp_down >=8 or tmp_right >=8:
                        return False
                    if self.free((tmp_down, tmp_right)) is False:  
                        return False
                return True
        return False


    def check(self, col: Color) -> bool:
        """Checks if given player is endangered in that position
        Parameters
        ----------
        col: Color
            Player to be checked
        
        Returns
        ----------
        bool
            True -- player endangered
            False -- player is not endangered
        """
        king: figure.Figure
        for item_i in self.s:
            for item in item_i:
                if item is not None and item.color == col and item.name == "king":
                    king = item                                                         #we found our king :)
        for item_i in self.s:
            for item in item_i:
                if item is not None and item.color != col:
                    if self.can_move((king.down, king.right), item):
                        return True
                    #return True
        return False
    
    def move(self, pos: tuple[int, int], fig: figure.Figure) -> bool:
        """Moves figure to position
        Parameters
        ----------
        
        
        Returns
        ----------
        bool
            True -- player endangered
            False -- player is not endangered
        """


    def avoid_check(self, col: Color) -> bool:
        """Checks if given player can avoid check
        Parameters
        ----------
        col: Color
            Player to be checked
        
        Returns
        ----------
        bool
            True -- player endangered
            False -- player is not endangered
        """
            

if __name__=="__main__":
    b = Board("board.state")