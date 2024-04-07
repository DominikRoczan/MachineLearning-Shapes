import tensorflow as tf
from tensorflow.keras import layers, models

# Zdefiniuj model SSD
def create_ssd_model(input_shape, num_classes):
    # Encoder
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    base_out = base_model.output

    # Dodaj dodatkowe warstwy do detekcji obiektów
    x = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(base_out)
    x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Detekcja obiektów
    classes_out = layers.Conv2D(num_classes * 4, (3, 3), padding='same')(x)
    locs_out = layers.Conv2D(num_classes * 4, (3, 3), padding='same')(x)

    model = models.Model(inputs=base_model.input, outputs=[classes_out, locs_out])
    return model

# Ustawienia modelu
input_shape = (224, 224, 3)  # Przykładowy rozmiar obrazu, dostosuj do swoich potrzeb
num_classes = 2  # Liczba klas, w tym przypadku pomieszczenia i szachty instalacyjne

# Ścieżki do danych treningowych, walidacyjnych i testowych
train_data_path = 'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/apartments/train'
val_data_path = 'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/apartments/val'
test_data_path = 'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/apartments/test'

# Stwórz model SSD
model = create_ssd_model(input_shape, num_classes)

# Skompiluj model
model.compile(optimizer='adam', loss='mse')

# Wyświetl architekturę modelu
model.summary()


# Wczytanie danych treningowych
train_dataset_rooms = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    class_names=['room', 'shaft'])  # Załóżmy, że 'room' to kategoria pokoi, a 'shaft' to kategoria szachtów

# Wczytanie danych walidacyjnych
val_dataset_rooms = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_path,
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    class_names=['room', 'shaft'])

# Wczytanie danych testowych
test_dataset_rooms = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_path,
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    class_names=['room', 'shaft'])

# Trenowanie modelu
model.fit(train_dataset_rooms, validation_data=val_dataset_rooms, epochs=10)

# Ewaluacja modelu na danych testowych
print(model.evaluate(test_dataset_rooms))
