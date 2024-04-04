import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Wczytaj wytrenowany model
model_path = 'mobilenet+2_Classe.h5'
model = load_model(model_path)

# Wczytaj obraz
from image_path import image_path

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
class_labels = ['Circle', 'Triangle']

threshold_probability = 0.5  # Próg prawdopodobieństwa do progowania
class_index = 1  # Indeks klasy trójkąta

# Dokonaj rozpoznawania kształtów


# Przeprowadź predykcję za pomocą modelu
predictions = model.predict(np.expand_dims(cv2.resize(image, (224, 224)), axis=0) / 255.0)
class_index = np.argmax(predictions)
probability = predictions[0][class_index]

# Jeżeli rozpoznano kształt, drukuj informacje
if probability > 0.65:

    # Wypisz index klasy i prawdopodobieństwo
    print(f'Rozpoznano kształt o indexie: {class_index} z prawdopodobieństwem: {probability:.2f}')
    # Wyświetl etykietę klasy
    print(f'Rozpoznany kształt: {class_labels[class_index]}')

    # Dodaj etykietę klasy na obrazie wraz z prawdopodobieństwem
    text = f'{class_labels[class_index]}, Prob: {probability:.2f}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

print(40 * '*')

# Wyświetl obraz z zaznaczonymi obszarami i podpisanymi kształtami
cv2.imshow('Classifier shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
