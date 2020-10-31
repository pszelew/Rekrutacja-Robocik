from common.vector3 import Vector3
from boat import Boat
import time

def run(t: float, n: int, url: str):
    assert t > 0, "n powinno byc > 0"
    assert n >= 5, "n powinno byc rowne co najmniej 5"
    i = 0
    in_pos = Vector3(0,0,0)
    boat = Boat(pos=in_pos, url=url)
    print("Starting boat! Let's dive")
    for i in range(n):
        boat.rand_move()
        print(f"Sending pos! {boat.pos.x}, {boat.pos.y}, {boat.pos.z}")
        boat.send_pos()
        time.sleep(t)
        i+=1
    boat.end_journey()
    print("Sending end!")



