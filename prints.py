<<<<<<< HEAD



def results(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
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
    print("Class Names:", class_names)
=======
from train import result_file_path, model_name


def results(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
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
    print("Class Names:", class_names)
>>>>>>> acf6128c69a0297734e9fe79087dc8bcf096cec3
