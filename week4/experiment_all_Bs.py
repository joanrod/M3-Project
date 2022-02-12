import datetime
import os

# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import timeit
# --------------------------------------------------Global parameters--------------------------------------------------
from tensorflow.python.keras.callbacks import EarlyStopping
start = timeit.default_timer()

f = open("env.txt", "r")
ENV = f.read().split('"')[1]

plot = True  # If plot is true, the performance of the model will be shown (accuracy, loss, etc.)
# backbone = 'EfficientNetB2_2'
num_of_experiment = 'testB'

# Paths to database
if ENV == "local":
    path_data = '../../MIT_small_train_1'
else:
    path_data = '/home/mcv/m3/datasets/MIT_small_train_1'

train_data_dir= path_data + '/train'
val_data_dir= path_data + '/validation'
test_data_dir= path_data + '/test'

# Image params
img_width = 224
img_height=224

# NN params
batch_size=16
number_of_epoch=50
LR = 0.001
momentum = 0.3
optim = 'Adagrad'

train_samples = 400
validation_samples= 800
test_samples = 800
# Experiment 2
freeze_layers = False  # If this variable is activated, we will freeze the layers of the base model to train parameters
# Experiment 4
new_layers = False  # Activate this variable to append new layers in between of the base model and the prediction layer

# TO DO...:
# CALLBACKS
# HYPERPARAMETER SEARCH
# DATA AUGMENTATION
# TRY THE 4 DATASETS OF MIT_SMALL_TRAIN
# mas propuestas...

# ---------------------------------------------------------------------------------------------------------------------

# Create the specific folders for this week, only if they don't exist
if not os.path.exists("models"):
    os.mkdir("models")

date_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
path_model = "models/models_exp_Bs/" + backbone + '_' + date_start
if not os.path.exists(path_model):
    os.mkdir(path_model)
    os.mkdir(path_model + "/results")
    os.mkdir(path_model + "/saved_model")

# Experiment 2
freeze_layers = False  # If this variable is activated, we will freeze the layers of the base model to train parameters

# Experiment 4
new_layers = False  # Activate this variable to append new layers in between of the base model and the prediction layer

# Store description of experiment setup in a txt file
with open(path_model + '/setup_description.txt', 'w') as f:
    f.write('Experiment set-up for: ' + path_model)
    f.write('\nExperiment number: ' + num_of_experiment)
    f.write('\nBackbone: ' + backbone)
    f.write('\nFreze Layers: ' + str(freeze_layers))
    f.write('\nBatch Norm + Relu: '+ str(new_layers))
    f.write('\nOptimizer: ' + optim)
    f.write('\nLearning Rate: ' + str(LR))
    f.write('\nTrain samples: ' + str(train_samples))
    f.write('\nValidation samples: ' + str(validation_samples))
    f.write('\nTest samples: ' + str(test_samples))
    f.write('\nBatch Size: ' + str(batch_size))


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 114.702
        x[:, :, 0] = x[:, :, 0] / 59.177
        x[1, :, :] -= 115.349
        x[1, :, :] = x[1, :, :] / 55.31
        x[2, :, :] -= 108.492
        x[2, :, :] = x[2, :, :] / 56.792
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 114.702
        x[:, :, 0] = x[:, :, 0] / 59.177
        x[:, :, 1] -= 115.349
        x[1, :, :] = x[1, :, :] / 55.31
        x[:, :, 2] -= 108.492
        x[2, :, :] = x[2, :, :] / 56.792
    return x
    
# create the base pre-trained model

base_model = backbone[current_backbone]

# file = path_model + "/saved_model" + '/completeModel.png'
# plot_model(base_model, to_file=file, show_shapes=True, show_layer_names=True)

x = base_model.layers[-2].output

if new_layers:
    print('Appending new layers to the model after the base_model...')
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = Dense(1024, activation='relu', name='extra_relu_1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = Dense(256, activation='relu', name='extra_relu_2')(x)
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)
print(model.summary())
with open(path_model + '/setup_description.txt', 'a') as f:
    f.write('\n\n Model Summary: \n' + model.summary())

# file = path_model + "/saved_model" + '/OurModel.png'
# plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)

# Freeze the layers from the model and train only the added ones
if freeze_layers:
    print('Freezing layers from the model and training only the added ones...')
    for layer in base_model.layers:
        layer.trainable = False

opt = tf.keras.optimizers.Adagrad(learning_rate=LR)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
# for layer in model.layers:
#     print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
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

history=model.fit(train_generator,
        steps_per_epoch=int(train_samples//batch_size),
        epochs=number_of_epoch,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps= int(validation_samples//batch_size),
        callbacks=[
            #'/path_model + "/saved_model/"+backbone_epoch{epoch:02d}_acc{val_accuracy:.2f}'+backbone+'.h5'
            ModelCheckpoint(path_model + "/saved_model/"+backbone+'.h5',
                monitor='val_accuracy',
                save_best_only=True,
            save_weights_only=True),
            CSVLogger(path_model+'/results/log_classification_'+ backbone + '_exp_' + num_of_experiment +'.csv', append=True, separator=';'),
            TensorBoard(path_model+'/tb_logs_'+ backbone + '_exp_' + num_of_experiment, update_freq=1),
            EarlyStopping(monitor='val_accuracy',patience=10,min_delta=0.001,mode='max')])
stop = timeit.default_timer()
print('Time: ', stop - start)

result = model.evaluate(test_generator)
print(result)


# list all data in history

# if plot:
#   # summarize history for accuracy
#   plt.plot(history.history['accuracy'])
#   plt.plot(history.history['val_accuracy'])
#   plt.title('model accuracy')
#   plt.ylabel('accuracy')
#   plt.xlabel('epoch')
#   plt.legend(['train', 'validation'], loc='upper left')
#   plt.savefig(path_model + '/results/accuracy.jpg')
#   plt.close()
#   # summarize history for loss
#   plt.plot(history.history['loss'])
#   plt.plot(history.history['val_loss'])
#   plt.title('model loss')
#   plt.ylabel('loss')
#   plt.xlabel('epoch')
#   plt.legend(['train', 'validation'], loc='upper left')
#   plt.savefig(path_model + '/results/loss.jpg')
