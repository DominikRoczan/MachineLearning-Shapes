import cv2
import numpy as np

# Wczytaj obraz
image_path = '4.jpg'
image = cv2.imread(image_path)

# Konwertuj obraz na odcienie szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detekcja krawędzi za pomocą operatora Canny
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Znajdź kontury w obrazie
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Przechowuj trójkąty i koła
triangles = []
circles = []

# Analizuj kontury w celu znalezienia obszarów trójkątów i kół
for contour in contours:
    # Aproksymuj kontur wielokątem
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Sprawdź, czy to trójkąt
    if len(approx) == 3:
        triangles.append(approx)
    # Sprawdź, czy to koło
    elif len(approx) > 8:  # Koło będzie miało więcej niż 8 wierzchołków
        circles.append(approx)

# Narysuj prostokątne obwiednie wokół trójkątów i podpisz pod nimi
for triangle in triangles:
    x, y, w, h = cv2.boundingRect(triangle)
    cv2.polylines(image, [triangle], True, (0, 255, 0), 2)
    cv2.putText(image, 'Triangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

# Narysuj prostokątne obwiednie wokół kół i podpisz pod nimi
for circle in circles:
    (x, y), radius = cv2.minEnclosingCircle(circle)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(image, center, radius, (0, 0, 255), 2)
    cv2.putText(image, 'Circle', (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                cv2.LINE_AA)

# Wydrukuj liczbę trójkątów i kół
print("Liczba trójkątów:", len(triangles))
print("Liczba kół:", len(circles))

# Wyświetl obraz z zaznaczonymi trójkątami i kołami
cv2.imshow('Detected Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
