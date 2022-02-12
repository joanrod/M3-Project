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
from customLoggerCallback import CustomLogger


def create_model(hp):

    kernel_size = hp.Choice('kernel', [3, 5])
    model = Sequential()
    model.add(layers.Conv2D(32, (kernel_size, kernel_size), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    if hp.Choice('BatchNorm1', [True]):
        model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Choice('dropout1', [0.1, 0.3, 0.5])))

    model.add(layers.Conv2D(32, (kernel_size, kernel_size), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    if hp.Choice('BatchNorm2', [True]):
        model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Choice('dropout2', [0.1, 0.3, 0.5])))

    if hp.Choice('GlobalAverage', [True]):
        model.add(layers.GlobalAvgPool2D())
    else:
        model.add(layers.Flatten())
    model.add(layers.Dense(hp.Choice('FC', [64, 128, 256]), activation='relu'))

    model.add(layers.Dropout(hp.Choice('dropoutFC', [0.1, 0.3, 0.5])))
    model.add(layers.Dense(8, activation='softmax'))
    print("Params: ", model.count_params())
    print(model.summary())
    optimizer = get_optimizer(hp.Choice('optimizer', ['Adagrad']), hp.Choice('learning_rate', [1e-2, 1e-3]))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main():
    f = open("env.txt", "r")
    ENV = f.read().split('"')[1]

    plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
    backbone = 'CustomCNN1'
    num_of_experiment = '0'

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
    batch_size=16
    number_of_epoch=50
    random_trials = 100

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
    # 'val_accuracy'
    keras_tuner.Objective("params", direction="max")
    hypertuner = RandomSearch(create_model, objective = 'val_accuracy', max_trials = random_trials, executions_per_trial=1, overwrite=True, directory="hp_search",
    project_name=backbone + '_exp_' + num_of_experiment + "_" + date_start + "_hp")
    hypertuner.search_space_summary()

    hypertuner.search(train_generator, steps_per_epoch=train_samples // batch_size, epochs=number_of_epoch,
                 validation_data=validation_generator, callbacks=[
                    # EarlyStopping(monitor='val_accuracy',patience=10,min_delta=0.001,mode='max'),
                    TensorBoard(path_model + '/tb_logs', update_freq=1)])

    # Retrieve the best model.
    best_model = hypertuner.get_best_models(num_models=1)[0]

    hypertuner.results_summary()

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(test_generator)

    print(f'loss: {loss}')
    print(f'accuracy: {accuracy}')


if __name__ == '__main__':
    main()