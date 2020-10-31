from __future__ import annotations
import math

class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"
    def dist(self, sec_vec: Vector3) -> float:
        res = math.sqrt((self.x - sec_vec.x)**2 
                + (self.y - sec_vec.y)**2
                + (self.z - sec_vec.z)**2)
        return res
                    
