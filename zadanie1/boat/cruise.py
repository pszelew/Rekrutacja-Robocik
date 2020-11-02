from vector3 import Vector3
from boat import Boat
import time
""" Starts our wonderful journey through seas 
        Parameters
        ----------
        t: float
            Period of generating new positions
        n: int
            How many moves before and of journey
        url: str
            Adress of server collecting data from boat
"""
def run(t: float, n: int, url: str):
    assert t > 0, "n powinno byc > 0"
    assert n >= 5, "n powinno byc rowne co najmniej 5"
    i = 0
    in_pos = Vector3(0,0,0)
    boat = Boat(pos=in_pos, url=url)
    # Create boat objects
    print("Inicjuje lodz! Zaczynamy!")
    for i in range(n):
        boat.rand_move()
        print(f"Wysylam pozycje! {boat.pos.x}, {boat.pos.y}, {boat.pos.z}")
        try:
            boat.send_pos()
        except:
            print("Nie udalo sie wyslac pozycji! Serwer nie odpowiada?")
        time.sleep(t)
        i+=1
    boat.end_journey()
    print("Koncze podroz!")



