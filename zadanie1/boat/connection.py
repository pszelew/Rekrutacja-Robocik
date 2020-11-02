import requests
import json

class Connection:
    """
    A class used to represent a connection to a server

    Attributes
    ----------
    url : str
        Adress of server we want to send data to

    Methods
    -------
    send(message: dict)
        Send message to the server
    """
    def __init__(self, url: str):
        """
        Parameters
        ----------
        url : str
            Adress of the server
        """
        self.url = url
    def send(self, message: dict):
        """ Send message to the server
        Parameters
        ----------
        message : dict
            Message to be sent
        """
        requests.post(url=self.url, data=json.dumps(message))
        # Send data