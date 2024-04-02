# tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
import os
from datetime import datetime

from prints import prints


def path_datasets():
    """"Path for data"""
    base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/TRvsCI'
    # base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/images'

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    return train_dir, val_dir, test_dir


def data_generator():
    """Data generators configuration"""
    train_datagen = (ImageDataGenerator
                     (rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                      shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
                      ))
    val_datagen = (ImageDataGenerator(rescale=1. / 255))
    test_datagen = (ImageDataGenerator(rescale=1. / 255))
    return train_datagen, val_datagen, test_datagen


def data_processing(train_dir, val_dir, test_dir, class_mode='binary', batch_size=16):
    """Data collection and processing"""

    train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        train_dir,
        target_size=(
            224, 224),
        classes=[
            'circles',
            'triangles'],
        batch_size=batch_size,
        shuffle=True,
        class_mode=class_mode)
    val_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(val_dir,
                                                                                                    target_size=(
                                                                                                        224, 224),
                                                                                                    classes=['circles',
                                                                                                             'triangles'],
                                                                                                    batch_size=batch_size,
                                                                                                    shuffle=False,
                                                                                                    class_mode=class_mode)
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(test_dir,
                                                                                                     target_size=(
                                                                                                         224, 224),
                                                                                                     classes=['circles',
                                                                                                              'triangles'],
                                                                                                     batch_size=batch_size,
                                                                                                     shuffle=False,
                                                                                                     class_mode=class_mode)

    return train_generator, val_generator, test_generator, class_mode, batch_size


base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Nazwa modelu
model_name = base_model.name
model_name = model_name[:-11]

# Zamrożenie wag modelu bazowego
base_model.trainable = False

# Budowa model
model = Sequential([
    base_model,
    layers.Conv2D(8, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((1, 1)),
    layers.BatchNormalization(),

    layers.Conv2D(16, (2, 2), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(32, (16, 16), activation='sigmoid', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (16, 16), activation='relu', padding='same'),
    layers.MaxPooling2D((1, 1)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(16, activation='relu'),

    # layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')
])
# model.summary()

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=BinaryCrossentropy(),
              metrics=['accuracy']
              )

# Folder na wyniki
result_folder = 'results'
os.makedirs(result_folder, exist_ok=True)

# Nazwa pliku tekstowego
result_file_name = f"{model_name}+2Classes+++working.txt"
result_file_path = os.path.join(result_folder, result_file_name)

# Directory TensorBoard
log_dir = (f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/'
           f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')
# log_dir = (f'E:/USERS/dominik.roczan/PycharmProjects/logs/'
#          f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')

os.makedirs(log_dir, exist_ok=True)

# TensorBoard Callback
tensorboard_train = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)

# Directory Checkpoint
checkpoint_filepath = 'results/checkpoint.model.keras'


# Wywołanie tensorboard w konsoli: tensorboard --logdir=C:/USERS/domin/OneDrive/Pulpit/Python/logs
# Wywołanie tensorboard w konsoli: tensorboard --logdir=E:/USERS/dominik.roczan/PycharmProjects/logs


def model_checkpoint_callback():
    return ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )


def early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        mode="min",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
        verbose=1
    )


def train_model(train_generator, val_generator, batch_size=16):
    # Początek czasu treningu
    start_time = datetime.now()

    # Trening modelu
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n // batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.n // batch_size,
                        epochs=1,
                        callbacks=[tensorboard_train, model_checkpoint_callback(), early_stopping()]
                        )

    # The model (that are considered the best) can be loaded as
    best_model = load_model('results/checkpoint.model.keras')

    # Zapis modelu do pliku .h5
    model.save(f'{model_name}+2_Classes++working.h5')
    best_model.save(f'best.{model_name}+2_Classes++working5.h5')

    # Podsumowanie modelu
    best_model.summary()

    # Koniec czasu treningu
    end_time = datetime.now()
    training_duration = end_time - start_time
    save_model = datetime.now()
    print(best_model)
    return training_duration, save_model, best_model


def data_evaluation(test_generator, best_model):
    # Pobierz dane do oceny modelu
    y_true = test_generator.labels  # prawdziwe etykiety z generatora danych
    y_pred = np.argmax(best_model.predict(test_generator), axis=1)  # przewidziane etykiety modelu
    y_pred_binary = (best_model.predict(test_generator) > 0.5).astype(int)

    return y_true, y_pred, y_pred_binary


def model_evaluation(y_true, y_pred_binary):
    # Ocena modelu
    classification_rep = classification_report(y_true, y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)

    return classification_rep, conf_matrix, accuracy


if __name__ == "__main__":
    train_dir, val_dir, test_dir = path_datasets()

    train_datagen, val_datagen, test_datagen = data_generator()
    data_processing(train_dir, val_dir, test_dir, class_mode='binary', batch_size=16)
    train_generator, val_generator, test_generator, class_mode, batch_size = data_processing(train_dir, val_dir,
                                                                                             test_dir,
                                                                                             class_mode='binary',
                                                                                             batch_size=16)
    early_stop = early_stopping()
    checkpoint = model_checkpoint_callback()

    training_duration, save_model, best_model = train_model(train_generator, val_generator, batch_size)

    y_true, y_pred, y_pred_binary = data_evaluation(test_generator, best_model)

    classification_rep, conf_matrix, accuracy = model_evaluation(y_true, y_pred_binary)

    prints(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
           classification_rep, conf_matrix, accuracy)
