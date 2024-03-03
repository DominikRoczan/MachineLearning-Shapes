from PIL import Image, ImageDraw
import random
import os
import shutil



def generuj_tlo_z_trojkatami(ilosc_jpg):
    szerokosc = 224
    wysokosc = 224

    for i in range(ilosc_jpg):
        tlo = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        obraz = Image.new('RGB', (szerokosc, wysokosc), color=tlo)
        draw = ImageDraw.Draw(obraz)

        ilosc_trojkatow = random.randint(1, 5)

        for _ in range(ilosc_trojkatow):
            punkty = []
            for _ in range(3):
                x = random.randint(0, szerokosc)
                y = random.randint(0, wysokosc)
                punkty.append((x, y))

            wypelnienie = random.choice(
                [None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])

            draw.polygon(punkty, fill=wypelnienie, outline=wypelnienie)

        obraz.save(f'tlo_z_trojkatami_{i + 1}.jpg', format='JPEG' )



    return ilosc_trojkatow

generuj_tlo_z_trojkatami(1)

