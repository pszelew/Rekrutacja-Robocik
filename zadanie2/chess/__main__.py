import cruise
import sys
if __name__ == "__main__":
    try:
        t = float(sys.argv[1])
        n = int(sys.argv[2])
        url = str(sys.argv[3])
    except:
        print("Nie udalo sie wczytac parametrow uruchomienia!")
        print("Skladnia:")
        print("1) t: float -- okres aktualiazcji")
        print("2) n: int -- ilosc ruchow >= 5")
        print("3) url: str -- 'adres_na_ktory_wysylane_maja_byc_dane':port")
        exit()
    cruise.run(t, n, url)
