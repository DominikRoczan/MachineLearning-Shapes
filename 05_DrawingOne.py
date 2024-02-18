import os

import cv2
import numpy as np

image_path = ('new_Image/X/23.jpg')
image = cv2.imread(image_path)

# Konwertuj obraz na odcienie szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Progowanie Otsu
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Zastosuj algorytm Canny do detekcji krawędzi
edges = cv2.Canny(thresholded_image, 50, 150)

# Znajdź kontury w obrazie po detekcji krawędzi
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

draw_image = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 3)

# Wyświetl obraz z zaznaczonymi obszarami i podpisanymi kształtami`

cv2.imshow(f'{image_path[16:]}', draw_image)

contours = len(contours)
print(f'All contours: {contours}')
cv2.waitKey(0)
cv2.destroyAllWindows()
