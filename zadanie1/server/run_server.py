from server import Server

def run(url: str, port: int) -> bool:
    """ Starts server 
        Parameters
        ----------
        url: str
            Adress of server collecting data from boat
        port: int
            Port to run server
    """
    serv = Server(url, port)
    # Create server
    serv.run()
    # Start server
        