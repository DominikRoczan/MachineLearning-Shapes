from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG19, VGG16, MobileNetV2, MobileNetV3Small, InceptionV3, InceptionResNetV2, ResNet152,
    DenseNet121, NASNetMobile, EfficientNetV2M)


from keras.utils import to_categorical



from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split
import numpy as np
import os
from datetime import datetime


# Ścieżka do katalogu głównego z obrazami
base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/images'

# Ścieżki do zbiorów
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Konfiguracja generatorów danych
train_datagen = (ImageDataGenerator
                 (rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'))

val_datagen = (ImageDataGenerator(rescale=1. / 255))
test_datagen = (ImageDataGenerator(rescale=1. / 255))

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='categorical',
                                                    color_mode='rgb')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), class_mode='categorical', color_mode='rgb')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), class_mode='categorical',
                                                  color_mode='rgb')
# Przekształć etykiety klas na wektory one-hot encoding
train_labels_one_hot = to_categorical(train_generator.labels, num_classes=2)
# Ładowanie modelu:
# base_model = InceptionV3(input_shape=(224,224, 3), include_top=False,weights='imagenet')
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model = InceptionResNetV2(input_shape=(224, 224, 3), include_top=False,weights='imagenet')

# base_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False,weights='imagenet')
# base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False,weights='imagenet')

# base_model = VGG19(input_shape=(224, 224, 3), include_top=False,weights='imagenet')
# base_model = VGG16(input_shape=(224, 224, 3), include_top=False,weights='imagenet')
# base_model = ResNet152(input_shape=(224, 224, 3), include_top=False,weights='imagenet')
# base_model = EfficientNetV2M(input_shape=(224, 224, 3), include_top=False,weights='imagenet')

# Nazwa modelu
model_name = base_model.name

# Zamrożenie wag modelu bazowego
base_model.trainable = False

# Budowa model
'''model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])'''

model = Sequential([
    base_model,
    layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # layers.BatchNormalization(),
    layers.Conv2D(32, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # layers.BatchNormalization(),
    layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.MaxPooling2D((1, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(2, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              # loss='categorical_crossentropy',
              metrics=['accuracy'])

# Utwórz podfolder w głównym katalogu "images" na wyniki
result_folder = 'results'
os.makedirs(result_folder, exist_ok=True)

# Nazwa pliku tekstowego
result_file_name = f"{model_name}.txt"
result_file_path = os.path.join(result_folder, result_file_name)

# Nazwa katalogu TensorBoard
# log_dir = f'E:/USERS/dominik.roczan/PycharmProjects/logs{model_name}'
# log_dir = f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}'

# Katalogi TensorBoard
log_dir_train = f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/train'
log_dir_val = f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/val'
log_dir_test = f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/test'

# log_dir_train = f'E:/USERS/dominik.roczan/PycharmProjects/logs{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/train'
# log_dir_test = f'E:/USERS/dominik.roczan/PycharmProjects/logs{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/test'
# log_dir_val = f'E:/USERS/dominik.roczan/PycharmProjects/logs{model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}/val'

os.makedirs(log_dir_train, exist_ok=True)
os.makedirs(log_dir_val, exist_ok=True)
os.makedirs(log_dir_test, exist_ok=True)

# TensorBoard Callback
tensorboard_train = TensorBoard(log_dir=log_dir_train, histogram_freq=0, write_graph=True, write_images=False)
tensorboard_val = TensorBoard(log_dir=log_dir_val, histogram_freq=0, write_graph=True, write_images=False)
tensorboard_test = TensorBoard(log_dir=log_dir_test, histogram_freq=0, write_graph=True, write_images=False)

# Wywołanie tensorboard w konsoli: tensorboard --logdir=C:/USERS/domin/OneDrive/Pulpit/Python/logs
# Wywołanie tensorboard w konsoli: tensorboard --logdir=E:/USERS/dominik.roczan/PycharmProjects/logs

# Początek czasu treningu
start_time = datetime.now()

# Trening modelu
model.fit(train_generator, epochs=32, callbacks=[tensorboard_train, tensorboard_val, tensorboard_test])

# Zapis modelu do pliku .h5
model.save(f'{model_name}.h5')

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
classification_rep = classification_report(y_true, y_pred, zero_division=1)
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
