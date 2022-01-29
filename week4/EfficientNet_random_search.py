import datetime
import os

import tensorflow as tf
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard


# --------------------------------------------------Global parameters--------------------------------------------------


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 114.702
        x[:, :, 0] = x[:, :, 0] / 59.177
        x[ 1, :, :] -= 115.349
        x[1, :, :] = x[ 1, :, :]/55.31
        x[ 2, :, :] -= 108.492
        x[2, :, :] = x[ 2, :, :] / 56.792
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 114.702
        x[:, :, 0] = x[:, :, 0]/59.177
        x[:, :, 1] -= 115.349
        x[1, :, :] = x[1, :, :] / 55.31
        x[:, :, 2] -= 108.492
        x[2, :, :] = x[2, :, :] / 56.792
    return x

def get_optimizer(opt, LR, mom):
    if opt == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=LR)
    elif opt == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR, momentum=mom)
    elif opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=mom)
    elif opt == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=LR)
    elif opt == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    elif opt == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax(lr=LR)
    elif opt == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(lr=LR)

    return optimizer

backbone = {
    'EfficientNetB0' : tf.keras.applications.EfficientNetB0(),
    'EfficientNetB1' : tf.keras.applications.EfficientNetB1(),
    'EfficientNetB2' : tf.keras.applications.EfficientNetB2(),
    'EfficientNetB3' : tf.keras.applications.EfficientNetB3()
}

activ_func_dict = {
    'relu' : tf.keras.activations.relu,
    'tanh' : tf.keras.activations.tanh,
    'elu' : tf.keras.activations.elu,
    'sigmoid' :tf.keras.activations.sigmoid
}

# Experiment 2
freeze_layers = False  # If this variable is activated, we will freeze the layers of the base model to train parameters

def create_model(hp):

    # create the base pre-trained model
    #base_model = backbone[hp.Choice('backbone', ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'])]
    base_model = backbone[hp.Choice('backbone', ['EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3'])]
    x = base_model.layers[-2].output

    if hp.Choice('new_layers', [True, False]):
        print('Appending new layers to the model after the base_model...')
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.5)(x)
        x = Dense(1024, activation='relu', name='extra_relu_1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.5)(x)
        x = Dense(256, activation='relu', name='extra_relu_2')(x)
    x = Dense(8, activation='softmax',name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    optimizer = get_optimizer(hp.Choice('optimizer', ['SGD', 'RMSprop', 'Adagrad', 'Adam']),
                                   hp.Choice('learning_rate', [1e-4, 1e-3, 1e-2, 1e-1]),
                                    hp.Choice('momentum', [0.2, 0.3, 0.6, 0.8, 0.9]))
                                   #hp.Choice('momentum', [0.0, 0.2, 0.3, 0.6, 0.8, 0.9]))
    #activ_func = hp.Choice('activation_function', ['relu', 'tanh', 'elu', 'sigmoid'])
    activ_func = hp.Choice('activation_function', ['relu', 'sigmoid'])

    for layer in model.layers:
        if str(layer.__class__.__name__) == "Activation":
            layer.activation = activ_func_dict[activ_func]

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_test(hp):

    # create the base pre-trained model
    base_model = backbone[hp.Choice('backbone', ['EfficientNetB0'])]
    x = base_model.layers[-2].output

    if hp.Choice('new_layers', [True, False]):
        print('Appending new layers to the model after the base_model...')
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.5)(x)
        x = Dense(1024, activation='relu', name='extra_relu_1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        x = Dense(256, activation='relu', name='extra_relu_2')(x)
    x = Dense(8, activation='softmax',name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    optimizer = get_optimizer(hp.Choice('optimizer', ['SGD']),
                                   hp.Choice('learning_rate', [1e-4]),
                                   hp.Choice('momentum', [0.0]))
    activ_func = hp.Choice('activation_function', ['relu'])

    for layer in model.layers:
        if str(layer.__class__.__name__) == "Activation":
            layer.activation = activ_func_dict[activ_func]

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main():
    f = open("env.txt", "r")
    ENV = f.read().split('"')[1]

    plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
    backbone = 'EfficientNetB2'
    num_of_experiment = '5'

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

    date_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_model = "models/" + backbone + '_exp_' + num_of_experiment + '_' + date_start
    if not os.path.exists(path_model):
        os.mkdir(path_model)
        os.mkdir(path_model + "/results")
        os.mkdir(path_model + "/saved_model")


    # Store description of experiment setup in a txt file
    with open(path_model + '/setup_description.txt', 'w') as f:
        f.write('Experiment set-up for: ' + path_model)
        f.write('\nExperiment number: ' + num_of_experiment)
        f.write('\nFreze Layers: ' + str(freeze_layers))
        f.write('\nTrain samples: ' + str(train_samples))
        f.write('\nValidation samples: ' + str(validation_samples))
        f.write('\nTest samples: ' + str(test_samples))
        f.write('\nBatch Size: ' + str(batch_size))
        f.write('\nEpochs: ' + str(number_of_epoch))
    hypertuner = RandomSearch(create_model, objective = 'val_accuracy', max_trials = 50, executions_per_trial=1, overwrite=True, directory="hp_search",
    project_name=backbone + '_exp_' + num_of_experiment + "_2")
    hypertuner.search_space_summary()

    hypertuner.search(train_generator, steps_per_epoch=400 // batch_size, epochs=number_of_epoch,
                 validation_data=validation_generator, callbacks=[
                    EarlyStopping(monitor='val_accuracy',patience=10,min_delta=0.001,mode='max'),
                    TensorBoard('/tb_logs', update_freq=1)])

    # Retrieve the best model.
    best_model = hypertuner.get_best_models(num_models=1)[0]

    hypertuner.results_summary()

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(test_generator)

    print(f'loss: {loss}')
    print(f'accuracy: {accuracy}')


if __name__ == '__main__':
    main()