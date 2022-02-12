import datetime
import os

import tensorflow as tf
from keras_tuner.tuners import RandomSearch
import keras_tuner
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from utils import *
from cnn_models import *

class CustomLogger(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("params: ", self.model.count_params())
  # def on_train_batch_begin(self, batch, logs=None):
  #   # print('Entrenamiento: batch {} comienza en {}'.format(batch, datetime.datetime.now().time()))
  #
  # def on_
  # train_batch_end(self, batch, logs=None):
  #   print('Entrenamiento: batch {} termina en {}'.format(batch, datetime.datetime.now().time()))
  #
  # def on_test_batch_begin(self, batch, logs=None):
  #   print('Evaluacion: batch {} comienza en {}'.format(batch, datetime.datetime.now().time()))
  #
  # def on_test_batch_end(self, batch, logs=None):
  #   print('Evaluacion: batch {} termina en {}'.format(batch, datetime.datetime.now().time()))