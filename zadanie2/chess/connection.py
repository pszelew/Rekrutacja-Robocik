import requests
import json

class Connection:
    def __init__(self, url: str):
        self.url = url
    def send(self, message: dict):
        requests.post(url=self.url, data=json.dumps(message))