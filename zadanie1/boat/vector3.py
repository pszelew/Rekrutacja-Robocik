from __future__ import annotations
import math

class Vector3:
    """
    A class used to represent a connection to 3D Vector

    Attributes
    ----------
    x : float
        Value of x-axis
    y : float
        Value of y-axis
    z : float
        Value of z-axis
        
    Methods
    -------
    dist(sec_vec: Vector3) -> float
        Return distance to the point
    """
    def __init__(self, x: float, y: float, z: float):
        """
        Parameters
        ----------
        x: float
            Value of x-axis
        y: float
            Value of y-axis
        z: float
            Value of z-axis
        """
        self.
        self.x = x
        self.y = y
        self.z = z
    def dist(self, sec_vec: Vector3) -> float:
        """ Return distance to the point
        Parameters
        ----------
        sec_vec: Vector3
            Second point of operation

        Returns
        -------
        float
            Distance to the point described by sec_vec
        """
        res: float
        res = math.sqrt((self.x - sec_vec.x)**2 
                + (self.y - sec_vec.y)**2
                + (self.y - sec_vec.y)**2)
        return res
                    
