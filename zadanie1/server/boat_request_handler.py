from http.server import BaseHTTPRequestHandler
import json
from vector3 import Vector3
import time

last_pos: Vector3
last_pos = None
last_time: float
last_time = None

class BoatHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    A class used to represent boat http handler

    Methods
    -------
    do_POST()
        Handle POST request
    """
    def do_POST(self):
        """Handle POST request
        """
        global last_pos
        global last_time
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        # Connection success
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        dic_data = json.loads(post_data)
        # Received data as json
        if "kill" in dic_data:
            # Kill received from boat
            print("Otrzymano sygnal kill. Wylaczanie serwera!")
            exit()
        cur_pos: Vector3
        cur_pos = Vector3(dic_data["pos_x"], dic_data["pos_y"], dic_data["pos_z"])
        cur_time: float
        cur_time = time.time()
        if last_pos is not None and last_time is not None:   
            cur_vel: float
            cur_vel = cur_pos.dist(last_pos)/(cur_time-last_time)
            print("Lodz osiagnela zawratna predkosc {:.2f} m/s!".format(cur_vel))
        else:
            print("Zbieram dane do obliczenia predkosci!!!")
            print("Prosze o chwile cierpliwosci")
        last_pos = cur_pos
        last_time = cur_time
        
        