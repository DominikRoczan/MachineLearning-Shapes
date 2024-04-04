import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_path import image_path

def load_image(image_path):
    """
    Load an image from the given path.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: Loaded image.
    """
    return cv2.imread(image_path)

def preprocess_image(image):
    """
    Preprocess the input image for prediction.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Preprocessed image.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image_rgb, (224, 224))

def predict_shape(image, model):
    """
    Predict the shape in the given image using the provided model.

    Parameters:
    image (numpy.ndarray): Input image.
    model: Pre-trained Keras model for prediction.

    Returns:
    tuple: Tuple containing predicted class index and probability.
    """
    image_input = np.expand_dims(image, axis=0)
    predictions = model.predict(image_input)
    class_index = np.argmax(predictions)
    probability = predictions[0][class_index]
    return class_index, probability

def display_result(image, class_index, probability, class_labels, probability_threshold=0.35):
    """
    Display the result of shape prediction on the image.

    Parameters:
    image (numpy.ndarray): Input image.
    class_index (int): Predicted class index.
    probability (float): Predicted probability.
    class_labels (list): List of class labels.
    probability_threshold (float): Probability threshold for classification. Default is 0.35.

    Returns:
    None
    """
    if probability > probability_threshold:
        print(f'Predicted shape with index: {class_index} and probability: {probability:.2f}')
        print(f'Predicted shape: {class_labels[class_index]}')

        # Add predicted class label to the image along with probability
        text = f'{class_labels[class_index]}, Prob: {probability:.2f}'
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

    print(40 * '*')

    # Display the image with annotated regions and labeled shapes
    cv2.imshow('Classifier shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load image
    image = load_image(image_path)

    # Load pre-trained model
    model_path = 'best.mobilenet+2_Classes++working5.h5'
    model = load_model(model_path)

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Predict shape
    class_index, probability = predict_shape(preprocessed_image, model)

    # Define class labels
    class_labels = ['Circle', 'Triangle']

    # Display result
    display_result(image, class_index, probability, class_labels)

if __name__ == "__main__":
    main()
