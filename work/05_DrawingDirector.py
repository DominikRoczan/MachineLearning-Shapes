import os
import cv2

# Ścieżka do folderu z obrazami
folder_path = '../new_Image/Train'

# Pobierz listę plików w folderze
files = os.listdir(folder_path)

# Iteruj przez każdy plik w folderze
for file in files:
    # Sprawdź, czy plik ma rozszerzenie obrazu (np. .jpg, .png, .jpeg)
    if file.endswith(('.jpg', '.jpeg', '.png')):
        # Wczytaj obraz
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)

        # Tutaj wykonaj operacje na obrazie, takie jak konwersja do odcieni szarości, detekcja krawędzi itp.
        # Na przykład:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresholded_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        draw_image = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 3)

        # Dodaj nazwę pliku do obrazu
        # cv2.putText(draw_image, file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5   , (0, 0, 255), 2)

        # Wyświetl obraz z zaznaczonymi konturami i nazwą pliku
        cv2.imshow(f'{file}', draw_image)
        cv2.waitKey(0)

# Po przetworzeniu wszystkich obrazów zakończ pętlę i zamknij wszystkie okna po wciśnięciu dowolnego klawisza
cv2.destroyAllWindows()
