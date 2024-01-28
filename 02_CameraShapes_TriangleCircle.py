import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Wczytaj wytrenowany model
model = load_model('inception_v3.h5')

# Funkcja do rozpoznawania figury na podstawie predykcji modelu
def predict_shape(image):
    # Przygotuj obraz do przekazania do modelu
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0

    # Dokonaj predykcji
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)

    # Zwróć indeks klasy
    return class_idx

# Wczytaj obraz
image_path = '3.jpg'
image = cv2.imread(image_path)

# Konwertuj obraz na odcienie szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Progowanie Otsu
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Zastosuj algorytm Canny do detekcji krawędzi
edges = cv2.Canny(thresholded_image, 50, 150)

# Znajdź kontury w obrazie po detekcji krawędzi
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Wypisz liczbę konturów
print('Liczba konturów:', len(contours))

# Dokonaj rozpoznawania figur
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]

    # Przeprowadź predykcję za pomocą modelu
    shape_idx = predict_shape(roi)

    # Pobierz nazwę kształtu z modelu
    class_names = list(model.class_indices.keys())
    shape_name = class_names[shape_idx]

    # Drukuj nazwę kształtu
    print(f'Znaleziony kształt: {shape_name}')

    # Narysuj prostokąt wokół obszaru detekcji
    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)

    # Dodaj nazwę kształtu na obrazie
    cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Wyświetl obraz z zaznaczonymi obszarami i podpisanymi kształtami
cv2.imshow('Detected Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
