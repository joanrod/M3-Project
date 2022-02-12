import datetime
import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from utils import *
from cnn_models import *

# Experiment 2
freeze_layers = False  # If this variable is activated, we will freeze the layers of the base model to train parameters

def create_model(num_blocks):
    if num_blocks == 1:
        model = customCNN1L()
    elif num_blocks == 2:
        model = customCNN2L()

    optimizer = get_optimizer('Adagrad', 0.01)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    f = open("env.txt", "r")
    ENV = f.read().split('"')[1]

    plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
    backbone = 'CustomCNN1L'
    num_blocks = 1
    num_of_experiment = '1'

    # Paths to database
    if ENV == "local":
        path_data = '../../MIT_small_train_1'
    else:
        path_data = '/home/mcv/m3/datasets/MIT_small_train_1'

    train_data_dir = path_data + '/train'
    val_data_dir = path_data + '/validation'
    test_data_dir = path_data + '/test'

    # Image params
    img_width = 224
    img_height = 224

    # NN params
    batch_size=32
    number_of_epoch=200


    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        preprocessing_function=preprocess_input,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None)

    train_generator = datagen.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    test_generator = datagen.flow_from_directory(test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = datagen.flow_from_directory(val_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    train_samples = 400
    validation_samples = 800
    test_samples = 800

    # ---------------------------------------------------------------------------------------------------------------------

    # Create the specific folders for this week, only if they don't exist
    if not os.path.exists("models"):
        os.mkdir("models")

    date_start = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    path_model = "models/" + backbone + "_" + num_of_experiment + '_' + date_start
    if not os.path.exists(path_model):
        os.mkdir(path_model)
        os.mkdir(path_model + "/results")
        os.mkdir(path_model + "/saved_model")


    # Store description of experiment setup in a txt file
    with open(path_model + '/setup_description.txt', 'w') as f:
        f.write('Experiment set-up for: ' + path_model)
        f.write('\nExperiment number: ' + num_of_experiment)
        f.write('\nBatch Size: ' + str(batch_size))
        f.write('\nEpochs: ' + str(number_of_epoch))
        f.write('\nEpochs: ' + str(number_of_epoch))

    model = create_model(num_blocks=num_blocks)
    model.summary()

    file = path_model + "/saved_model/model_arch.png"
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)

    history = model.fit(train_generator,
                        steps_per_epoch=int(train_samples // batch_size),
                        epochs=number_of_epoch,
                        shuffle=True,
                        validation_data=validation_generator,
                        validation_steps=int(validation_samples // batch_size),
                        callbacks=[
                            # '/path_model + "/saved_model/"+backbone_epoch{epoch:02d}_acc{val_accuracy:.2f}'+backbone+'.h5'
                            ModelCheckpoint(path_model + "/saved_model/" + backbone + '.h5',
                                            monitor='val_accuracy',
                                            save_best_only=True,
                                            save_weights_only=True),
                            CSVLogger(
                                path_model + '/results/log_classification_' + backbone + '_exp_' + num_of_experiment + '.csv',
                                append=True, separator=';'),
                            TensorBoard(path_model + '/tb_logs', update_freq=1),
                            # EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, mode='max')
                        ])

    result = model.evaluate(test_generator)
    print(result)

    print("compactness ratio: {}".format((result[1]/model.count_params()/100000)))

    plot_acc_and_loss(history, path_model)

if __name__ == '__main__':
    main()