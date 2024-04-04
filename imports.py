from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
# Callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
from datetime import datetime