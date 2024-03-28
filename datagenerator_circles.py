from PIL import Image, ImageDraw
import random
import os
import shutil

def generuj_tlo_z_kolami_i_okregami(szerokosc, wysokosc, ilosc_jpg_circle, destination_path_circles):
    # Funkcja generująca obrazy z losowymi kołami i okręgami
    for i in range(ilosc_jpg_circle):
        # Losowy kolor tła dla każdego obrazu
        tlo = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        obraz = Image.new('RGB', (szerokosc, wysokosc), color=tlo)
        draw = ImageDraw.Draw(obraz)

        # Losowa liczba kół i okręgów w obrazie
        # ilosc_kol = random.randint(1, 5)
        ilosc_kol = 1

        # Rysowanie kół i okręgów
        for _ in range(ilosc_kol):
            x1 = random.randint(0, szerokosc)
            y1 = random.randint(0, wysokosc)
            promien = random.randint(10, min(szerokosc, wysokosc) // 2)
            x2 = x1 + promien
            y2 = y1 + promien

            # Losowe wypełnienie i krawędź koła lub okręgu
            wypelnienie = random.choice([None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])
            draw.ellipse([x1, y1, x2, y2], fill=wypelnienie, outline=wypelnienie)

        # Zapisywanie obrazu
        obraz.save(os.path.join(destination_path_circles, f'CIRCLES_{i + 1}.jpg'), format='JPEG')

def generuj_wiele_obrazow(szerokosc, wysokosc, ilosc_jpg_circle, source_path):
    # Funkcja generująca wiele obrazów
    os.makedirs(source_path, exist_ok=True)
    generuj_tlo_z_kolami_i_okregami(szerokosc, wysokosc, ilosc_jpg_circle, source_path)

def przechowalnia(ilosc_jpg_circle, source_path, destination_path_circles):
    # Tworzenie katalogu docelowego
    os.makedirs(destination_path_circles, exist_ok=True)
    for i in range(ilosc_jpg_circle):
        source_file = f'CIRCLES_{i + 1}.jpg'
        try:
            shutil.copy(os.path.join(source_path, source_file), os.path.join(destination_path_circles, source_file))
        except shutil.SameFileError:
            # Ignorowanie błędu SameFileError
            pass

szerokosc = 224
wysokosc = 224
ilosc_jpg_circle =  40
source_path = 'Generator'
destination_path_circles = os.path.join(source_path, 'Circles')

generuj_wiele_obrazow(szerokosc, wysokosc, ilosc_jpg_circle, destination_path_circles)
przechowalnia(ilosc_jpg_circle, destination_path_circles, destination_path_circles)
