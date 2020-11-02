import http.server
import socketserver
import signal
import sys
from boat_request_handler import BoatHTTPRequestHandler



class Server():
    """
    A class used to represent a server

    Attributes
    ----------
    url: str
        Adress of server
    port: int
        Port of server
    handler: BoatHTTPRequestHandler
        Handler for events

    Methods
    -------
    send_pos()
        Send position through connection manager
    rand_move()
        Do random move in scene coordinates (absolute) [-5:5, -5:0, -5:5]
    end_journey()
        Send kill signal to server
    """
    def __init__(self, url: str, port: int):
        """
        Parameters
        ----------
        url: str
            Adress of server
        port: int
            Port of server
        """
        self.url = url
        self.port = port
        self.handler = BoatHTTPRequestHandler
    def run(self):
        """ Start server"""
        with socketserver.TCPServer(("", self.port), self.handler) as httpd:
            print("Start serwera na", self.url+":"+str(self.port))
            httpd.allow_reuse_address = True
            # Configuration of server
            httpd.serve_forever()
            # Start server
            