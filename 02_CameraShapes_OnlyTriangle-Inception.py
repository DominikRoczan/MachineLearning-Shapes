import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Wczytaj wytrenowany model
model = load_model('mobilenetv2_1.00_224.h5')


# Funkcja do rozpoznawania trójkątów na podstawie predykcji modelu
def predict_triangle(image):
    # Przygotuj obraz do przekazania do modelu
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0

    # Dokonaj predykcji
    predictions = model.predict(image)

    # Sprawdź, czy model rozpoznał trójkąt
    is_triangle = predictions[0][1] > 0.25  # Próg prawdopodobieństwa

    return is_triangle, predictions[0][1]


# Wczytaj obraz
image_path = '3.jpg'
image = cv2.imread(image_path)

# Przeprowadź predykcję za pomocą modelu dla całego obrazu
is_triangle, probability = predict_triangle(image)

# Zaznacz trójkąt w ramce, dodaj podpis i wyświetl prawdopodobieństwo
if is_triangle:
    x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'Triangle, Proba: {probability:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
else:
    cv2.putText(image, 'Not found.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Wyświetl obraz z zaznaczonym trójkątem i informacją o prawdopodobieństwie
cv2.imshow('Detected Triangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
