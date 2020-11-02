from vector3 import Vector3
from connection import Connection
import random

class Boat:
    """
    A class used to represent a boat

    Attributes
    ----------
    pos : Vector3
        3D Vector containing current submarine position
    con : Connection
        Connection manager

    Methods
    -------
    send_pos()
        Send position through connection manager
    rand_move()
        Do random move in scene coordinates (absolute) [-5:5, -5:0, -5:5]
    end_journey()
        Send kill signal to server
    """
    def __init__(self, pos: Vector3, url: str):
        """
        Parameters
        ----------
        pos: Vector3
            Starting position of boat
        url: str
            Adress of the server
        """
        self.pos = pos                  
        self.con = Connection(url)
        random.seed()
        # Random seed of random number generator
    def send_pos(self):
        """Send position through connection manager"""
        self.con.send({"pos_x": self.pos.x, "pos_y": self.pos.y, "pos_z": self.pos.z})
    def rand_move(self):
        """Do random move in scene coordinates (absolute) [-5:5, -5:0, -5:5]"""
        rand_pos: Vector3
        rand_pos.x = random.uniform(-5.0, 5.0)
        # Random move [-5.0, 5.0]
        rand_pos.y = random.uniform(-5.0, 0)
        # Random move [-5.0, 0]
        rand_pos.z = random.uniform(-5.0, 5.0)
        # Random move [-5.0, 5.0]
        self.pos = rand_pos
    def end_journey(self):
        """Send kill signal to server"""
        self.con.send({"kill": 9})


