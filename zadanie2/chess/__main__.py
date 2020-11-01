from chess import app
import sys
if __name__ == "__main__":
    try:
        file = str(sys.argv[1])
    except:
        print("Nie udalo sie wczytac parametrow uruchomienia!")
        print("Skladnia:")
        print("1) file: str -- nazwa pliku z stanem do wczytania")
        exit()
    app.run(file)
