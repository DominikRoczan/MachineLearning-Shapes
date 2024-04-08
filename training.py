# tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications import MobileNetV2
from keras.models import Sequential, load_model
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

from prints import results


def path_datasets():
    """
    Returns paths for train, validation, and test datasets.

    Returns:
    train_dir (str): Path to the training dataset directory.
    val_dir (str): Path to the validation dataset directory.
    test_dir (str): Path to the test dataset directory.
    """

    base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/TRvsCI'
    # base_dir = f'G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/images'

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    return train_dir, val_dir, test_dir


def data_generator():
    """
    Returns configuration for data generators.

    Returns:
    train_datagen: Configuration for the training data generator.
    val_datagen: Configuration for the validation data generator.
    test_datagen: Configuration for the test data generator
    """

    train_datagen = (ImageDataGenerator
                     (rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                      shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
                      ))
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return train_datagen, val_datagen, test_datagen


def data_processing(train_dir, val_dir, test_dir, class_mode='binary', batch_size=16):
    """
    Collection and process the data.

    Args:
    train_dir (str): Path to the training dataset directory.
    val_dir (str): Path to the validation dataset directory.
    test_dir (str): Path to the test dataset directory.
    class_mode (str): One of "binary" or "categorical". Default is "binary".
    batch_size (int): Batch size for training. Default is 16.

    Returns:
    train_generator: Generator for training data.
    val_generator: Generator for validation data.
    test_generator: Generator for test data.
    class_mode (str): Type of classes for classification.
    batch_size (int): Batch size used.
    """

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


def get_base_model():
    """
    Get the base model for transfer learning.

    Returns:
    model_name (str): Name of the base model.
    base_model: Pre-trained MobileNetV2 model.
    """

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Nazwa modelu
    model_name = base_model.name
    model_name = model_name[:-11]

    # Zamrożenie wag modelu bazowego
    base_model.trainable = False
    return model_name, base_model


def build_model(base_model):
    """
    Build the custom model on top of the base model.

    Args:
    base_model: Pre-trained base model.

    Returns:
    model: Compiled model.
    """

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
    return model


def compile_model(model):
    """
    Compile the model with specified optimizer, loss function, and metrics.

    Args:
    model: Model to compile.

    Returns:
    model: Compiled model.
    result_file_path (str): Path to the result file.
    """

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy']
                  )

    # Folder na wyniki
    result_folder = 'results'
    os.makedirs(result_folder, exist_ok=True)

    # Nazwa pliku tekstowego
    result_file_name = f"{model_name}+2_Classes.txt"
    result_file_path = os.path.join(result_folder, result_file_name)

    return model, result_file_path


def callback_tensorboard():
    """
    Configure TensorBoard and return its callback.

    Returns:
    tensorboard_train: TensorBoard callback.
    """

    log_dir = (f'C:/USERS/domin/OneDrive/Pulpit/Python/logs/'
                f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')
    #log_dir = (f'E:/USERS/dominik.roczan/PycharmProjects/logs/'
     #          f'{model_name}....{datetime.now().strftime("%Y.%m.%d....%H.%M")}')

    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard Callback
    tensorboard_train = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)

    # Directory Checkpoint
    # checkpoint_filepath = 'results/checkpoint.model.keras'

    return tensorboard_train


# Wywołanie tensorboard w konsoli: tensorboard --logdir=C:/USERS/domin/OneDrive/Pulpit/Python/logs
# Wywołanie tensorboard w konsoli: tensorboard --logdir=E:/USERS/dominik.roczan/PycharmProjects/logs

def model_checkpoint_callback():
    """
    Configure and return model checkpoint callback.

    Returns:
    ModelCheckpoint callback.
    """

    return ModelCheckpoint(
        filepath='results/checkpoint.model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )


def early_stopping():
    """
    Configure and return early stopping callback.

    Returns:
    EarlyStopping callback.
    """
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


def train_model(train_generator, val_generator, epochs = 15, batch_size=16):
    """
    Train the model, save it and calculate the training time.

    Args:
    train_generator: Generator for training data.
    val_generator: Generator for validation data.
    batch_size (int): Batch size for training. Default is 16.

    Returns:
    training_duration: Duration of the training process.
    save_model: Timestamp when the model was saved.
    best_model: Best trained model.
    :param epochs:
    """

    # Start of training time
    start_time = datetime.now()

    # Model training
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n // batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.n // batch_size,
                        epochs=epochs,
                        callbacks=[tensorboard_train, model_checkpoint_callback(), early_stopping()]
                        )

    # The model (that are considered the best) can be loaded as
    best_model = load_model('results/checkpoint.model.keras')

    # Save the model to .h5 file
    model.save(f'{model_name}+2_Classes.h5')
    best_model.save(f'best.{model_name}+2_Classes.h5')

    # Model summary
    best_model.summary()

    # End of training time
    end_time = datetime.now()
    training_duration = end_time - start_time
    save_model = datetime.now()

    return training_duration, save_model, best_model


def data_evaluation(test_generator, best_model):
    """
    Evaluate the model on test data.

    Args:
    test_generator: Generator for test data.
    best_model: Best trained model.

    Returns:
    y_true: True labels.
    y_pred: Predicted labels.
    y_pred_binary: Predicted labels in binary format.
    """


    y_true = test_generator.labels  # true labels from the data generator
    y_pred = np.argmax(best_model.predict(test_generator), axis=1)  # predicted labels of the model
    y_pred_binary = (best_model.predict(test_generator) > 0.5).astype(int)

    return y_true, y_pred, y_pred_binary


def model_evaluation(y_true, y_pred_binary):
    """
    Evaluate the model using classification report and confusion matrix.

    Args:
    y_true: True labels.
    y_pred_binary: Predicted labels in binary format.

    Returns:
    classification_rep: Classification report.
    conf_matrix: Confusion matrix.
    """

    classification_rep = classification_report(y_true, y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    # accuracy = accuracy_score(y_true, y_pred_binary)

    return classification_rep, conf_matrix

'''def results(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
            classification_rep, conf_matrix):
    """
    Save evaluation results to a text file.

    Args:
        best_model (keras.Model): Trained model with best performance.
        train_generator (keras.utils.Sequence): Generator for training data.
        val_generator (keras.utils.Sequence): Generator for validation data.
        test_generator (keras.utils.Sequence): Generator for test data.
        training_duration (datetime.timedelta): Duration of the training process.
        save_model (datetime.datetime): Timestamp when the model was saved.
        classification_rep (str): Classification report generated by model evaluation.
        conf_matrix (numpy.ndarray): Confusion matrix generated by model evaluation.

    Returns:
        None
    """

    train_results = best_model.evaluate(train_generator),
    print(f'Dawaj: {train_results}')
    loss_train_results = round(train_results[0][0], 4),
    acc_train_results = round(train_results[0][1], 4),
    print("Train Loss:", loss_train_results),
    print("Train Accuracy:", acc_train_results)

    # Validation set results
    val_results = best_model.evaluate(val_generator)
    loss_val_results = round(val_results[0], 4)
    acc_val_results = round(val_results[1], 4)
    print(f'Dawaj_2: {val_results}')
    print("Validation Loss:", loss_val_results)
    print("Validation Accuracy:", acc_val_results)

    # Test set results
    test_results = best_model.evaluate(test_generator)
    print(f'Dawaj_3: {test_results}')
    loss_test_results = round(test_results[0], 4)
    acc_test_results = round(test_results[0], 4)
    print("Test Loss:", loss_test_results)
    print("Test Accuracy:", acc_test_results)

    # Writing results to a .txt file
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

    # Dictionary mapping class labels to indices
    class_indices = train_generator.class_indices
    print("Class Indices:", class_indices)

    # The reverse dictionary
    indices_to_classes = {v: k for k, v in class_indices.items()}
    print("Indices to Classes:", indices_to_classes)

    # A list of class names
    class_names = list(class_indices.keys())
    print("Class Names:", class_names)'''




if __name__ == "__main__":
    """
    Main entry point of the script.
    """

    # Loads paths to the datasets
    train_dir, val_dir, test_dir = path_datasets()

    # Configures data generators
    # Processes data and creates data generators


    # train_datagen, val_datagen, test_datagen = data_generator()
    train_generator, val_generator, test_generator, class_mode, batch_size = data_processing(train_dir,
                                                                                             val_dir,
                                                                                             test_dir,
                                                                                             class_mode='binary',
                                                                                             batch_size=16)

    # Retrieves the base model for transfer learning
    # Builds the custom model on top of the base model
    # Compiles the model with specified optimizer, loss function, and metrics
    model_name, base_model = get_base_model()
    model = build_model(base_model)
    model, result_file_path = compile_model(model)

    # Sets up callbacks for TensorBoard, ModelCheckpoint, and EarlyStopping
    tensorboard_train = callback_tensorboard()
    early_stop = early_stopping()
    model_checkpoint_callback()

    # Trains the model and evaluates its performance
    training_duration, save_model, best_model = train_model(train_generator, val_generator, epochs=5, batch_size=batch_size)
    y_true, y_pred, y_pred_binary = data_evaluation(test_generator, best_model)

    # Prints classification report and confusion matrix
    classification_rep, conf_matrix = model_evaluation(y_true, y_pred_binary)
    results(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
            classification_rep, conf_matrix)