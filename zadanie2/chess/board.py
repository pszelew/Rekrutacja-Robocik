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
import copy
class Board:
    """
    A class used to represent a board from the game of chess

    Attributes
    ----------
    s: list[figures.figure.Figure]
        List keeping state of game
    dic_ver: dict
        Dictionary of relation: in_game coordinates --> real board. Vertical axis
    dic_hor: dict
        Dictionary of relation: in_game coordinates --> real board. Vertical axis
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
        self.dic_ver: dict
        self.dic_hor: dict
        self.dic_hor = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}
        self.dic_ver = {0: "8", 1: "7", 2: "6", 3: "5", 4: "4", 5: "3", 6: "2", 7: "1"}
        with open(file) as f:
            raw_data = f.read()
        raw_data = raw_data.replace("[", "").replace("]", ",")
        raw_data = raw_data.replace("'", "").replace("\n", "")
        lst_data = raw_data.split(",")
        for index, item in enumerate(lst_data):
            lst_data[index] = item.strip() 
        lst_data = lst_data[:-1]
        self.s = [lst_data[i*8:i*8+8] for i in range(8)]
        # self.s as multi dim table
        # Maybe not efficient but it is easier to visualise for me
        temp_color: Color
        temp_class: figure.Figure
        # Magic tools we are going to use later
        for i, item_i in enumerate(self.s):
            for j, item in enumerate(item_i):
                if item == "--":
                    # Empty space                                    
                    self.s[i][j] = None
                    continue
                if item[0] == "b":
                    # Color of figure is black
                    temp_color = Color.BLACK
                if item[0] == "w":
                    # Color of figure is white
                    temp_color = Color.WHITE
                if item[1] == "p":
                    # Chess pawn
                    temp_class = pawn.Pawn
                elif item[1] == "r":
                    # Chess rook
                    temp_class = rook.Rook
                elif item[1] == "k":
                    # Chess knight
                    temp_class = knight.Knight
                elif item[1] == "b":
                    # Chess bishop
                    temp_class = bishop.Bishop
                elif item[1] == "q":
                    # Chess quenn
                    temp_class = queen.Queen
                elif item[1] == "W":
                    # Chess king
                    temp_class = king.King
                self.s[i][j] = temp_class(temp_color, i, j)
                # Create objects
    def __str__(self):
        """
        """
        out_str = ""
        for i in range(8):
            out_str += "["
            for j in range(8):
                out_str += "'"
                if self.s[i][j] is None:
                    out_str += "--"
                else:
                    if self.s[i][j].color is Color.WHITE:
                        out_str += "w"
                    else:
                        out_str += "b"
                    out_str += self.s[i][j].short
                out_str += "'"
                if j < 7:
                    out_str += ", "
                else:
                    out_str += "]"
            out_str += "\n"
        return out_str

    def free(self, pos: tuple[int, int]) -> bool:
        """Checks if position is free
        Parameters
        ----------
        pos: tuple[int, int]
            Position to check [down, right]
        
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
        """Checks if figure can move to given position.
        Parameters
        ----------
        pos: tuple[int, int]
            Position to check [down, right]
        fig: figure.Figure
            Figure to be checked
        Returns
        ----------
        bool
            True -- figure can move there
            False -- figure can not move there
        """
        
        if self.free(pos) is False and self.s[pos[0]][pos[1]].color is fig.color:
            # occupied by our own figure
            # we cant move and capture figure. There is nothing to look for
            return False
    
        to_move: tuple[int, int]
        to_move = (pos[0]-fig.down, pos[1]-fig.right)
        
        if to_move == (0, 0):
            return False
        for mov in fig.pos_moves:
            if mov.down == to_move[0] and mov.right == to_move[1]:
                # it is a perfect move. It's super effective!          
                return True
            if mov.it == False:
                # it failed. It's not effective                                             
                continue
            f: int
            if mov.down is not 0:
                # how many iterations to reach target                                             
                f = to_move[0]//mov.down
            else:                                                                               
                f = to_move[1]//mov.right
                # how many iterations to reach target                                                         
            if f > 0 and f * mov.down == to_move[0] and f * mov.right == to_move[1]:
                #if it is possible to reach dest through iter   
                for i in range(f - 1):                                                            
                    tmp_down: int
                    tmp_right: int
                    tmp_down = fig.down+(i + 1) * mov.down
                    tmp_right = fig.right + (i + 1) * mov.right
                    if tmp_down > 7 or tmp_right > 7 or tmp_down < 0 or tmp_right < 0:
                        # exceeds dimensions of board
                        # move will not succeed
                        return False
                    if self.free((tmp_down, tmp_right)) is False:
                        # we encountered on our way another figure and we cant jump through it  
                        return False
                # We reached space before without problem
                # 1) There is enemy figure and we can capture it
                # 2) Place ahead is free and we can take it
                return True
        return False

    def find_king(self, col: Color) -> king.King:
        """Give me back my king! Returns king of desired color
        Parameters
        ----------
        col: Color
            Player to be checked
        
        Returns
        ----------
        king.King
            Object describing our king
        """
        my_king: king.King
        # King in current move
        my_king = None
        for item_i in self.s:
            # Iterate through first dimension of multi dim list
            for fig in item_i:
                # Iterate through second dimension of multi dim list
                if fig is not None and fig.color == col and fig.name == "king":
                    # We found our king :)  
                    my_king = fig
        return my_king


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
        my_king: king.King
        my_king = self.find_king(col)
        if my_king is None:
            raise Exception("King of desired color not found")
                                                            
        for item_i in self.s:
            # Iter through 1st dim of our multi dim list
            for fig in item_i:
                # Iter through 2nd dim of our multi dim list
                if fig is not None and fig.color != col:
                    if self.can_move((my_king.down, my_king.right), fig):
                        # There is enemy figure that can capture us there
                        return True
        return False
    
    def move(self, pos: tuple[int, int], fig: figure.Figure) -> bool:
        """Moves figure to position
        Parameters
        ----------
        pos: tuple[int, int]
            Position we want move to [down, right]
        fig: figure.Figure
            Figure to be moved
        Returns
        ----------
        bool
            True -- Success
            False -- It failed
        """
        if self.can_move(pos, fig):
            # We can move there
            self.s[fig.down][fig.right] = None
            # We remove pawn from previous place
            self.s[pos[0]][pos[1]] = fig
            # Change board state
            fig.down = pos[0]
            fig.right = pos[1]
            return True
        # We can't reach desired place
        return False
    def avoid_check(self, col: Color) -> bool:
        """Checks if given player can avoid check
        Parameters
        ----------
        col: Color
            Player to be checked
        
        Returns
        ----------
        bool
            True -- player can avoid check
            False -- player can not avoid check
        """
        if self.check(col) is False:
            #there is co check condition. What should we avoid then?
            return True

        # First option --> move king somewhere. Its most common way
        # It is subproblem of second option so we exclude it there
        org_king: king.King
        temp_king: king.King
        temp_board: Board

        org_king = self.find_king(col)
        if org_king is None:
            raise Exception("King of desired color not found")
        # Find king of desired color
        for move in org_king.pos_moves:
            # Iterate through possible king move
            temp_king = copy.deepcopy(org_king)
            temp_board = copy.deepcopy(self)
            # Deep copies of original king and board objects
            temp_down: int
            temp_right: int
            temp_down = move.down + temp_king.down
            temp_right = move.right + temp_king.right
            # Check move to those positions
            if temp_down > 7 or temp_right > 7 or temp_down < 0 or temp_right < 0:
                # Exceeds dimensions of board
                continue
            if temp_board.can_move((temp_down, temp_right), temp_king):
                # If our king can move to a place
                temp_board.move((temp_down, temp_right), temp_king)
                # Move temp king to desired position
                if temp_board.check(col):
                    # If enemy still has high ground
                    continue
                    # Just try something else
                return True
                # Wow it worked, we are fine and safe

        # Second option --> block with another figure
        # Lets use some cpu, simle way:
        for i, item_i in enumerate(self.s):
            for j, fig in enumerate(item_i):
                # Iterate through our multi dim list describing state of board
                if fig is None:
                    # If there is no figure
                    continue
                if fig.color is not col:
                    # If it isn't our figure. We can't move it
                    continue
                if fig.name is "king":
                    # We already checked it
                    continue
                for k in range(8):
                    for l in range(8):
                        # It looks awfull:) For in for in for in for
                        # But it checks if any move can end with a block
                        if self.can_move((k, l), fig):
                            temp_fig: figure.Figure
                            temp_fig = copy.deepcopy(fig)
                            temp_board = copy.deepcopy(self)
                            # Copy board and our figure to avoid messing up with original
                            temp_board.move((k, l), temp_fig)
                            if temp_board.check(col) is False:
                                return True                       
        return False
        
    def check_mate(self, col: Color) -> list[tuple[int, int, int, int]]:
        """Checks possibilities of winning by chosen player
        Parameters
        ----------
        col: Color
            Player to be checked
        
        Returns
        ----------
        list[tuple[int, int, int, int]]
            Move that gives us winning condition (start_down, start_right, end_down, end_right)
        """
        enemy_col: Color
        if col is Color.WHITE:
            enemy_col = Color.BLACK
        else:
            enemy_col = Color.WHITE
        lst: list
        lst = []

        # 1) We already checked our enemy, its over if it is our move
        if self.check(enemy_col):
            lst.append((-1, -1, -1, -1))
            return lst
        # 2) We have to do something to win

        for i, item_i in enumerate(self.s):
            for j, fig in enumerate(item_i):
                # Iterate through our multi dim list describing state of board
                if fig is None:
                    # If there is no figure
                    continue
                if fig.color is not col:
                    # If it isn't our figure. We can't move it
                    continue
                for k in range(8):
                    for l in range(8):
                        # It looks awfull:) For in for in for in for
                        # But it checks if any move can end with a block
                        if self.can_move((k, l), fig):
                            temp_fig: figure.Figure
                            temp_fig = copy.deepcopy(fig)
                            temp_board = copy.deepcopy(self)
                            # Copy board and our figure to avoid messing up with original
                            temp_board.move((k, l), temp_fig)
                            # Now check if we checked our opponent
                            if temp_board.check(enemy_col) is False:
                                # Move gives us nothing
                                continue
                            # Now we know that enemy is checked. Can he escape?  
                            if temp_board.avoid_check(enemy_col) is False:
                                lst.append((fig.down, fig.right, k, l))
        return lst

            

if __name__=="__main__":
    b = Board("board.state")