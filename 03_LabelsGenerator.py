from work._Work import *

# Pobranie nazw klas
class_names = list(class_indices.keys())

# Przewidywanie klas i prawdopodobieństw dla danych walidacyjnych i testowych
predictions_val = model.predict(val_generator)
predictions_test = model.predict(test_generator)

# Iteracja po danych walidacyjnych i ich przewidywaniach
for i, (image_batch_val, label_batch_val) in enumerate(val_generator):
    # Przerwanie pętli po wszystkich wsadach walidacyjnych
    if i >= len(val_generator):
        break

    # Iteracja po każdym elemencie w wsadzie walidacyjnym
    for j, (image_val, label_val) in enumerate(zip(image_batch_val, label_batch_val)):
        # Przewidywanie dla danego obrazu
        prediction_val = predictions_val[i * val_generator.batch_size + j]
        predicted_class_val = np.argmax(prediction_val)
        predicted_class_name_val = class_names[predicted_class_val]

        # Prawdopodobieństwa dla przewidywanej klasy
        probability_val = prediction_val[predicted_class_val]

        # Nazwa obrazu
        image_name_val = val_generator.filenames[i * val_generator.batch_size + j]

        # Wydrukuj nazwę obrazu, jego predykcję i prawdopodobieństwo
        print(f"|--VALID--|....{image_name_val}....PRED: {predicted_class_name_val}....PROBA: {probability_val:.4f}")

# Iteracja po danych testowych i ich przewidywaniach
for i, (image_batch_test, label_batch_test) in enumerate(test_generator):
    # Przerwanie pętli po wszystkich wsadach testowych
    if i >= len(test_generator):
        break

    # Iteracja po każdym elemencie w wsadzie testowym
    for j, (image_test, label_test) in enumerate(zip(image_batch_test, label_batch_test)):
        # Przewidywanie dla danego obrazu
        prediction_test = predictions_test[i * test_generator.batch_size + j]
        predicted_class_test = np.argmax(prediction_test)
        predicted_class_name_test = class_names[predicted_class_test]

        # Prawdopodobieństwa dla przewidywanej klasy
        probability_test = prediction_test[predicted_class_test]

        # Nazwa obrazu
        image_name_test = test_generator.filenames[i * test_generator.batch_size + j]

        # Wydrukuj nazwę obrazu, jego predykcję i prawdopodobieństwo
        print(f"|--TEST--|....{image_name_test}....PRED: {predicted_class_name_test}....PROBA: {probability_test:.4f}")
