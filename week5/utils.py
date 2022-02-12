from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf


activ_func_dict = {
    'relu' : tf.keras.activations.relu,
    'tanh' : tf.keras.activations.tanh,
    'elu' : tf.keras.activations.elu,
    'sigmoid' :tf.keras.activations.sigmoid
}

def get_optimizer(opt, LR, mom=0):
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

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        # Returns the default image data format convention: A string, either 'channels_first' or 'channels_last'.
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


def plot_acc_and_loss(history, path_model, retrained=False):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if retrained:
        file = path_model + '/results' + '/accuracyRetrained.jpg'
    else:
        file = path_model + '/results' + '/accuracy.jpg'
    plt.savefig(file)
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if retrained:
        file = path_model + '/results' + '/lossRetrained.jpg'
    else:
        file = path_model + '/results' + '/loss.jpg'
    plt.savefig(file)
    plt.close()


def plot_acc_and_loss_all(history1, history2, history3, history4, path_model, retrained=False):
    # summarize history for accuracy
    plt.plot(history1['accuracy'], label="accuracy_mitSplit1")
    plt.plot(history1['val_accuracy'], label="val_accuracy_mitSplit1")
    plt.plot(history2['accuracy'], label="accuracy_mitSplit2")
    plt.plot(history2['val_accuracy'], label="val_accuracy_mitSplit2")
    plt.plot(history3['accuracy'], label="accuracy_mitSplit3")
    plt.plot(history3['val_accuracy'], label="val_accuracy_mitSplit3")
    plt.plot(history4['accuracy'], label="accuracy_mitSplit4")
    plt.plot(history4['val_accuracy'], label="val_accuracy_mitSplit4")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    if retrained:
        file = path_model + '/accuracyRetrained.jpg'
    else:
        file = path_model + '/accuracy.jpg'
    plt.savefig(file)
    plt.close()

    # summarize history for loss
    plt.plot(history1['loss'], label="loss_mitSplit1")
    plt.plot(history1['val_loss'], label="val_loss_mitSplit1")
    plt.plot(history2['loss'], label="loss_mitSplit2")
    plt.plot(history2['val_loss'], label="val_loss_mitSplit2")
    plt.plot(history3['loss'], label="loss_mitSplit3")
    plt.plot(history3['val_loss'], label="val_loss_mitSplit3")
    plt.plot(history4['loss'], label="loss_mitSplit4")
    plt.plot(history4['val_loss'], label="val_loss_mitSplit4")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    if retrained:
        file = path_model + '/lossRetrained.jpg'
    else:
        file = path_model + '/loss.jpg'
    plt.savefig(file)
    plt.close()