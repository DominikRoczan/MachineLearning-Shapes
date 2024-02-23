import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

# Wczytaj model z pliku H5
model_path = 'mobilenet+3_Classes.h5'  # Zmień ścieżkę do twojego modelu
model = load_model(model_path)


img_path = '107.jpg'
imag = img_path

img_size = (224, 224)
# last_conv_layer_name = "conv2d_1"  # Zmień na nazwę warstwy konwolucyjnej w twoim modelu
last_conv_layer_name = "conv2d"  # Zmień na nazwę warstwy konwolucyjnej w twoim modelu


def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = mpl.cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    display(Image(cam_path))


img_array = preprocess_input(get_img_array(imag, size=img_size))

# Generuj heatmapę
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Zapisz i wyświetl Grad-CAM
save_and_display_gradcam(imag, heatmap)



# Wyświetl heatmapę
plt.matshow(heatmap)
plt.show()
