import run_server
import sys
if __name__ == "__main__":
    try:
        url = str(sys.argv[1])
        port = int(sys.argv[2])
    except:
        print("Nie udalo sie wczytac parametrow uruchomienia!")
        print("Skladnia:")
        print("1) url: str -- 'adres_na_ktory_wysylane_maja_byc_dane'")
        print("2) port: int -- port na ktorym uruchomiono serwer")
        exit()
    run_server.run(url, port)