import cv2
from ultralytics import YOLO

from _ImagePath import image_path

# Wczytaj zdjęcie za pomocą OpenCV
image = cv2.imread(image_path)

# Utwórz instancję modelu YOLOv8
model = YOLO('yolov8n.pt')

# Wykonaj detekcję obiektów na zdjęciu
results = model(image)

# Sprawdź, czy są jakieś wykryte obiekty
if isinstance(results, list) and results:
    # Iteruj przez listę wykrytych obiektów
    for detection in results[0]:
        # Upewnij się, że wystarczająca ilość elementów jest dostępna w wykryciu
        if len(detection) >= 6:
            class_id = int(detection[5])
            class_label = model.names[class_id]
            if class_label == 'bicycle':
                print("Znaleziono rower!")
                xmin, ymin, xmax, ymax = [int(i) for i in detection[:4]]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Wyświetl obraz z zaznaczonym rowerem (jeśli został wykryty)
    cv2.imshow('Zdjęcie z zaznaczonym rowerem', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie wykryto żadnych obiektów lub wynik detekcji nie jest poprawny.")
