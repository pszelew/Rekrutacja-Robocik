from common.vector3 import Vector3
from common.connection import Connection
import random

class Boat:
    def __init__(self, pos: Vector3, url: str):
        self.pos = pos
        self.con = Connection(url)
        random.seed()
    def send_pos(self):
        self.con.send({"pos_x": self.pos.x, "pos_y": self.pos.y, "pos_z": self.pos.z})
    def rand_move(self):
        rand_pos = Vector3(0,0,0)
        rand_pos.x = random.uniform(-5.0, 5.0)
        rand_pos.y = random.uniform(-5.0, 0)
        rand_pos.z = random.uniform(-5.0, 5.0)
        self.pos = rand_pos
    def end_journey(self):
        self.con.send({"kill": 9})


