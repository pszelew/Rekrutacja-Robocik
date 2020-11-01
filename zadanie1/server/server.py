import http.server
import socketserver
import signal
import sys


from boat_request_handler import BoatHTTPRequestHandler



class Server():
    def __init__(self, url: str, port: int):
        self.url = url
        self.port = port
        self.handler = BoatHTTPRequestHandler
    def run(self):
        with socketserver.TCPServer(("", self.port), self.handler) as httpd:
            print("Start serwera na", self.url+":"+str(self.port))
            httpd.allow_reuse_address = True
            httpd.serve_forever()
            