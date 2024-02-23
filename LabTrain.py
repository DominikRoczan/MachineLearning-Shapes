from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (MobileNetV2 )
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import os
from datetime import datetime

from _ImagePath import epochs

# Ścieżka do datasets
base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/images'

# Tworzenie ścieżek do zbiorów
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Konfiguracja generatorów danych
train_datagen = (ImageDataGenerator
                 (rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
                  ))
val_datagen = (ImageDataGenerator(rescale=1. / 255))
test_datagen = (ImageDataGenerator(rescale=1. / 255))

# Pobieranie i przetwarzanie danych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

# Ładowanie modelu:
# base_model = InceptionV3(input_shape=(224,224, 3), include_top=False,weights='imagenet')
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model = InceptionResNetV2(input_shape=(224, 224, 3), include_top=False,weights='imagenet')

# Nazwa modelu
model_name = base_model.name
model_name = model_name[:-11]

# Zamrożenie wag modelu bazowego
base_model.trainable = False

# Budowa model
model = Sequential([
    base_model,
    layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D((1, 1)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    # layers.Dropout(0.25),
    layers.Dense(3, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Początek czasu treningu
start_time = datetime.now()

# Trening modelu
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    epochs=epochs
                    )


# Pobierz dane do oceny modelu
y_true = train_generator.labels  # prawdziwe etykiety z generatora danych
y_pred = np.argmax(model.predict(train_generator), axis=1)  # przewidziane etykiety modelu

# Ocena modelu
classification_rep = classification_report(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Wyniki na zbiorze treningowym
train_results = model.evaluate(train_generator)
loss_train_results = round(train_results[0], 4)
acc_train_results = round(train_results[1], 4)
print("Train Loss:", loss_train_results)
print("Train Accuracy:", acc_train_results)

# Wyniki na zbiorze walidacyjnym
val_results = model.evaluate(val_generator)
loss_val_results = round(val_results[0], 4)
acc_val_results = round(val_results[1], 4)
print("Validation Loss:", loss_val_results)
print("Validation Accuracy:", acc_val_results)

# Wyniki na zbiorze testowym
test_results = model.evaluate(test_generator)
loss_test_results = round(test_results[0], 4)
acc_test_results = round(test_results[1], 4)
print("Test Loss:", loss_test_results)
print("Test Accuracy:", acc_test_results)
print(f'ACC: {accuracy_score}')

# Otrzymaj słownik przypisujący etykiety klas do indeksów
class_indices = train_generator.class_indices

# Wydrukuj słownik
print("Class Indices:", class_indices)

# Otrzymaj odwrotny słownik, przypisujący indeksy do etykiet
indices_to_classes = {v: k for k, v in class_indices.items()}

# Wydrukuj odwrotny słownik
print("Indices to Classes:", indices_to_classes)

# Otrzymaj listę klas
class_names = list(class_indices.keys())

# Wydrukuj listę klas
print("Class Names:", class_names)
