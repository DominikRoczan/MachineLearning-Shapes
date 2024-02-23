# skrypt do labelowania
# rescalowanie
# class mode = binary
# co on drukuje print(image)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (MobileNetV2, InceptionV3)
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
from datetime import datetime

# Ścieżka do datasets
base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/images'

# Tworzenie ścieżek do zbiorów
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


def custom_preprocessing(image):
    # Odjęcie 125 od wartości pikseli
    return image


# Konfiguracja generatorów danych
train_datagen = (ImageDataGenerator
                 (preprocessing_function=custom_preprocessing, rescale=1. / 255,
                  rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
                  ))

val_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1. / 255
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rescale=1. / 255
)
batch_size = 128

# Pobieranie i przetwarzanie danych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    # classes=['Circle', 'Not found', 'Triangle']
)

# Ładowanie modelu:
# base_model = InceptionV3(input_shape=(224,224, 3), include_top=False,weights='imagenet')
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Nazwa modelu
model_name = base_model.name
model_name = model_name[:-11]

# Zamrożenie wag modelu bazowego
base_model.trainable = False

# Budowa model
model = Sequential([
    base_model,
    # layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'),
    # layers.MaxPooling2D((2, 2)),
    # layers.BatchNormalization(),
    layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D((1, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.25),
    layers.Dense(3, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Folder na wyniki
result_folder = 'results'
os.makedirs(result_folder, exist_ok=True)

# Nazwa pliku tekstowego
result_file_name = f"{model_name}+3Classes.txt"
result_file_path = os.path.join(result_folder, result_file_name)

# Katalog TensorBoard
# log_dir = (f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/'
#            f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')
log_dir = (f'E:/USERS/dominik.roczan/PycharmProjects/logs/'
           f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')

os.makedirs(log_dir, exist_ok=True)

# TensorBoard Callback
tensorboard_train = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)

# Wywołanie tensorboard w konsoli: tensorboard --logdir=C:/USERS/domin/OneDrive/Pulpit/Python/logs
# Wywołanie tensorboard w konsoli: tensorboard --logdir=E:/USERS/dominik.roczan/PycharmProjects/logs

# Początek czasu treningu
start_time = datetime.now()

# Trening modelu
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    epochs=22,
                    callbacks=[tensorboard_train]
                    )

# Zapis modelu do pliku .h5
model.save(f'{model_name}+3_Classes.h5'),

# Podsumowanie modelu
model.summary()

# Koniec czasu treningu
end_time = datetime.now()
training_duration = end_time - start_time
save_model = datetime.now()

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

# Zapis wyników do pliku .txt
with open(result_file_path, 'w') as result_file:
    result_file.write("Model Name: {}\n".format(model_name))
    result_file.write("\n")
    result_file.write('Data: {}\n'.format(save_model))
    result_file.write("Training Duration: {}\n".format(training_duration))
    result_file.write("\n")
    result_file.write("Train Accuracy: {}\n".format(acc_train_results))
    result_file.write("Validation Accuracy: {}\n".format(acc_val_results))
    result_file.write("Test Accuracy: {}\n".format(acc_test_results))
    result_file.write("\n")
    result_file.write("Train loss: {}\n".format(loss_train_results))
    result_file.write("Validation loss: {}\n".format(loss_val_results))
    result_file.write("Test loss: {}\n".format(loss_test_results))
    result_file.write("\n\n")
    result_file.write("Classification Report:\n")
    result_file.write(classification_rep)
    result_file.write("\n\n")
    result_file.write("Confusion Matrix:\n")
    result_file.write(np.array2string(conf_matrix, separator=', '))

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
