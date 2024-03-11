from Shapes_01_Training import *

# Pobranie nazw klas
class_names = list(class_indices.keys())

# Przewidywanie klas i prawdopodobieństw dla danych walidacyjnych i testowych
predictions_val = model.predict(val_generator)
predictions_test = model.predict(test_generator)

directory_name = 'Labels'
os.makedirs(directory_name, exist_ok=True)
file_name = 'Labels.txt'
result_dir = os.path.join(directory_name, file_name)

# Iteracja po danych walidacyjnych i ich przewidywaniach

with open(result_dir, 'w') as file:
    file.write(f'Classe: {indices_to_classes}' + '\n')
    # file.write(f'Epochs: {epochs}' + '\n' + '\n')

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

            kod_val = f"|--VAL--|....{image_name_val}....PRED: {predicted_class_name_val}....PROBA: {probability_val:.4f}"
            file.write(kod_val + '\n')


directory_name_2 = 'Labels2'
os.makedirs(directory_name_2, exist_ok=True)

file_name2 = 'Labels2.txt'
result_dir2 = os.path.join(directory_name_2, file_name2)
# Iteracja po danych testowych i ich przewidywaniach
with open(result_dir2, 'w') as file:
    file.write(f'Classe: {indices_to_classes}' + '\n')
    # file.write(f'Epochs: {epochs}'+'\n'+'\n')

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

            print(
                f"|--TEST--|....{image_name_test}....PRED: {predicted_class_name_test}....PROBA: {probability_test:.4f}")


            kod_test = f"|--TEST--|....{image_name_test}....PRED: {predicted_class_name_test}....PROBA: {probability_test:.4f}"


            file.write(kod_test + '\n')
