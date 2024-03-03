import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Wczytaj wytrenowany model
model_path = 'mobilenet+2Classe.h5'
model = load_model(model_path)

# Wczytaj obraz
from _ImagePath import image_path

image = cv2.imread(image_path)

# Konwertuj obraz na przestrzeń kolorów RGB (OpenCV wczytuje obrazy w przestrzeni BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (224, 224))

# Dodaj dodatkowy wymiar do obrazu, aby pasował do wymagań modelu
image_input = np.expand_dims(image_resized, axis=0)
shape = image_input.shape
# Przeprowadź predykcję za pomocą wczytanego modelu
predictions = model.predict(image_input)

# Odczytaj wyniki predykcji
class_index = np.argmax(predictions)
probability = predictions[0][class_index]

# Odczytaj wyniki predykcji
class_index = np.argmax(predictions)
probability = predictions[0][class_index]

# Przykładowe etykiety klas
class_labels = ['Circle', 'Not found', 'Triangle']

threshold_probability = 0.5  # Próg prawdopodobieństwa do progowania
class_index = 2  # Indeks klasy trójkąta


# Zastosuj algorytm Canny do detekcji krawędzi
edges = cv2.Canny(image_rgb, 50, 150)

# Znajdź kontury w obrazie po detekcji krawędzi
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 20
filtered_contours = [
    contour for contour in contours if
    cv2.contourArea(contour) > min_contour_area
]



# Dokonaj rozpoznawania kształtów
if len(filtered_contours) > 0:
    for contour in filtered_contours:

        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y + h, x:x + w]
        # print(f'Roi: {roi}')

        # Przeprowadź predykcję za pomocą modelu
        predictions = model.predict(np.expand_dims(cv2.resize(roi, (224, 224)), axis=0) / 255.0)
        print(40 * '*')
        print('Predykcja kształtów: ', np.round(predictions, 3))
        class_index = np.argmax(predictions)
        print(20 * '*')
        print(f'Klasa: {class_index}')
        probability = predictions[0][class_index]
        print('Prawdopodobieństwo: ', round(probability, 2))

        # Jeżeli rozpoznano kształt, drukuj informacje
        if probability > 0.65:

            # Wypisz index klasy
            print(f'Rozpoznano kształt o indexie: {class_index} z prawdopodobieństwem: {probability:.2f}')

            # Dodaj index klasy na obrazie wraz z prawdopodobieństwem
            if class_index == 0:
                # Narysuj prostokąt wokół obszaru detekcji
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = f'Circle, Prob: {probability:.2f}'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif class_index == 1:
                # Narysuj prostokąt wokół obszaru detekcji
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = f'Not found, Prob: {probability:.2f}'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            elif class_index == 2:
                # Narysuj prostokąt wokół obszaru detekcji
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = f'Triangle, Prob: {probability:.2f}'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

else:
    print('Brak konturów znalezionych.')

print(40 * '*')
print(f'All contours: {len(contours)}')

# Wyświetl obraz z zaznaczonymi obszarami i podpisanymi kształtami
cv2.imshow('Detected Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
