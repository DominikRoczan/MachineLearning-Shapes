from PIL import Image, ImageDraw
import random
import os
import shutil

def generuj_tlo_z_trojkatami(szerokosc, wysokosc, ilosc_jpg, destination_path):
    # Funkcja generująca obrazy z losowymi trójkątami
    for i in range(ilosc_jpg):
        # Losowy kolor tła dla każdego obrazu
        tlo = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        obraz = Image.new('RGB', (szerokosc, wysokosc), color=tlo)
        draw = ImageDraw.Draw(obraz)

        # Losowa liczba trójkątów w obrazie
        ilosc_trojkatow = random.randint(1, 5)

        # Rysowanie trójkątów
        for _ in range(ilosc_trojkatow):
            punkty = []
            # Losowe współrzędne wierzchołków trójkąta
            for _ in range(3):
                x = random.randint(0, szerokosc)
                y = random.randint(0, wysokosc)
                punkty.append((x, y))

            # Losowe wypełnienie i krawędź trójkąta
            wypelnienie = random.choice([None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])
            draw.polygon(punkty, fill=wypelnienie, outline=wypelnienie)

        # Zapisywanie obrazu
        obraz.save(os.path.join(destination_path, f'TRIANGLES_{i + 1}.jpg'), format='JPEG')

def generuj_wiele_obrazow(szerokosc, wysokosc, ilosc_jpg, destination_path):
    # Funkcja generująca wiele obrazów
    os.makedirs(destination_path, exist_ok=True)
    generuj_tlo_z_trojkatami(szerokosc, wysokosc, ilosc_jpg, destination_path)

def przechowalnia(ilosc_jpg, source_path, destination_path):
    # Tworzenie katalogu docelowego
    os.makedirs(destination_path, exist_ok=True)
    for i in range(ilosc_jpg):
        source_file = f'TRIANGLES_{i + 1}.jpg'
        try:
            shutil.copy(os.path.join(source_path, source_file), os.path.join(destination_path, source_file))
        except shutil.SameFileError:
            # Ignorowanie błędu SameFileError
            pass



szerokosc = 224
wysokosc = 224
ilosc_jpg = 14
source_path = 'Generator'
destination_path = os.path.join(source_path, 'Triangles')

generuj_wiele_obrazow(szerokosc, wysokosc, ilosc_jpg, destination_path)
przechowalnia(ilosc_jpg, destination_path, destination_path)
