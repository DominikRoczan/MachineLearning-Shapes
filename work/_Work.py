'''https://github.com/nikhilroxtomar/Object-Detection-from-Scratch-
in-TensorFlow/blob/main/1%20-%20Simple%20Object%20Detection/train.py#L56'''


import cv2

predictions = [...]  # add your predictions here

for bounding_box in predictions['predictions']:
    x0 = bounding_box['x'] - bounding_box['width'] / 2
    x1 = bounding_box['x'] + bounding_box['width'] / 2
    y0 = bounding_box['y'] - bounding_box['height'] / 2
    y1 = bounding_box['y'] + bounding_box['height'] / 2

    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))
    cv2.rectangle(img, start_point, end_point, color=(0, 0, 0), thickness=1)

    cv2.putText(
        image,
        bounding_box["class"],
        (int(x0), int(y0) - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=2
    )

cv2.imwrite("example_with_bounding_boxes.jpg", image)