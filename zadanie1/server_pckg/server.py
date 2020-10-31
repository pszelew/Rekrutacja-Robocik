import http.server
import socketserver
from common.vector3 import Vector3
from common.boat_request_handler import BoatHTTPRequestHandler


class Server():
    def __init__(self, url: str, port: int):
        self.url = url
        self.port = port
        self.handler = BoatHTTPRequestHandler
    def run(self):
        with socketserver.TCPServer(("", self.port), self.handler) as httpd:
            print("serving at port", self.port)
            httpd.serve_forever()

        