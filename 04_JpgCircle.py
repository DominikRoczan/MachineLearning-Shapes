import cv2
import numpy as np

# Wczytaj obraz
image_path = '4.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detekcja krawędzi
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Zastosuj transformację Hougha do detekcji okręgów
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)

# Przechowuj znalezione klasy
found_circles = 0

if circles is not None:
    circles = np.uint16(np.around(circles))

    # Dokonaj rozpoznawania okręgów
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]

        # Sprawdź, czy kształt jest okręgiem
        if radius > 0:
            # Rysuj prostokątną obwódkę wokół okręgu
            x, y = center[0] - radius, center[1] - radius
            w, h = 2 * radius, 2 * radius
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Podpisz okrąg
            cv2.putText(image, 'Circle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            found_circles += 1

# Wyprintuj wyniki
print("Liczba okręgów:", found_circles)

# Wyświetl obraz z zaznaczonymi kształtami
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
