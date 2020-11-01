class PossibleMove():
    """
    Class describing one possible move for figure

    Attributes
    ----------
    down: int
        Move down
    right: int
        Move right
    it: bool
        Can figure iterate move?
    """
    def __init__(self, down: int, right: int, it: bool):
        """
        Parameters
        ----------
        down: int
            Move down
        right: int
            Move right
        it: bool
            Can figure iterate move?
        """
        self.down = down
        self.right = right
        self.it = it
