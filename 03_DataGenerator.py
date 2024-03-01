from PIL import Image, ImageDraw
import random
import os
import shutil


def generuj_tlo_z_trojkatami():
    # Tworzymy obraz o wymiarach 224x224 pikseli
    szerokosc = 224
    wysokosc = 224
    tlo = random.randint(0,255),random.randint(0,255),random.randint(0,255)
    obraz = Image.new('RGB', (szerokosc, wysokosc), color=tlo)
    draw = ImageDraw.Draw(obraz)

    # Losowa ilość trójkątów na obrazie (od 1 do 5)
    ilosc_trojkatow = random.randint(1, 5)

    for _ in range(ilosc_trojkatow):
        # Losowo wybieramy punkty trójkąta
        punkty = []
        for _ in range(3):
            x = random.randint(0, szerokosc)
            y = random.randint(0, wysokosc)
            punkty.append((x, y))

        # Losowo wybieramy, czy trójkąt będzie wypełniony
        wypelnienie = random.choice([None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])

        # Rysujemy trójkąt
        draw.polygon(punkty, fill=wypelnienie, outline=wypelnienie)

    # Zapisujemy obraz do pliku w formacie JPG
    obraz.save('tlo_z_trojkatami.jpg', format='JPEG')

    # Zwracamy ilość trójkątów
    return ilosc_trojkatow

# Przykładowe użycie
ilosc_trojkatow = generuj_tlo_z_trojkatami()
print("Wygenerowano", ilosc_trojkatow, "trójkątów na tle.")

def przechowalnia():
    directory_name = 'Generator'
    os.makedirs(directory_name, exist_ok=True)

    result_dir = os.path.join(directory_name, 'Triangles')
    os.makedirs(result_dir, exist_ok=True)

    # Ścieżka do pliku docelowego
    destination_path = os.path.join(result_dir, 'tlo_z_trojkatami.jpg')

    # Usuń plik, jeśli już istnieje
    if os.path.exists(destination_path):
        os.remove(destination_path)

    # Przenieś wygenerowany plik do katalogu wynikowego "Triangles"
    shutil.move('tlo_z_trojkatami.jpg', result_dir)

przechowalnia()