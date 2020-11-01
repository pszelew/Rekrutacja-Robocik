from http.server import BaseHTTPRequestHandler
import json
from vector3 import Vector3

last_pos = Vector3(0,0,0)

class BoatHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global last_pos
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        dic_data = json.loads(post_data)
        if "kill" in dic_data:
            print("Otrzymano sygnal kill. Wylaczanie serwera!")
            exit()
        pos = Vector3(dic_data["pos_x"], dic_data["pos_y"], dic_data["pos_z"])
        print("Lodz osiagnela zawratna predkosc {:.2f} m/s!".format(pos.dist(last_pos)))
        last_pos = pos
        
        