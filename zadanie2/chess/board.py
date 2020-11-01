
class Board:
    """
    A class used to represent a board from the game of chess

    Attributes
    ----------
    s : list
        List keeping state of game

    Methods
    -------
    template(attribute=None)
        something
    """
    def __init__(self, file: str):
        """
        Parameters
        ----------
        file : str
            The name of the file containing board state to be load
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

        
        
            

if __name__=="__main__":
    b = Board("board.state")