import keras
import numpy as np
from keras.models import Sequential
from keras import layers
import tensorflow as tf


backbone = {
    'EfficientNetB0' : tf.keras.applications.EfficientNetB0(),
    'EfficientNetB1' : tf.keras.applications.EfficientNetB1(),
    'EfficientNetB2' : tf.keras.applications.EfficientNetB2(),
    'EfficientNetB3' : tf.keras.applications.EfficientNetB3(),
    'EfficientNetB4' : tf.keras.applications.EfficientNetB4()
}



def customCNN1L():
    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.GlobalAvgPool2D())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(8, activation='softmax'))

    return model

def customCNN2L():
    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAvgPool2D())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(8, activation='softmax'))

    return model