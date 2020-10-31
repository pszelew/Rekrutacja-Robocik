from server import Server

def run(url: str, port: int) -> bool:
    serv = Server(url, port)
    serv.run()
        