def prints(best_model, train_generator, val_generator, test_generator, training_duration, save_model,
           classification_rep, conf_matrix, accuracy):
    """Save results to a text file."""

    train_results = best_model.evaluate(train_generator),
    loss_train_results = round(train_results[0], 4),
    acc_train_results = round(train_results[1], 4),
    print("Train Loss:", loss_train_results),
    print("Train Accuracy:", acc_train_results)

    # Wyniki na zbiorze walidacyjnym
    val_results = best_model.evaluate(val_generator)
    loss_val_results = round(val_results[0], 4)
    acc_val_results = round(val_results[1], 4)
    print("Validation Loss:", loss_val_results)
    print("Validation Accuracy:", acc_val_results)

    # Wyniki na zbiorze testowym
    test_results = best_model.evaluate(test_generator)
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
