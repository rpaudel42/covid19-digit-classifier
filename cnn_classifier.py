# ******************************************************************************
# cnn_classifier.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 4/18/20   Paudel     Initial version,
# ******************************************************************************
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import models
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
import matplotlib.cm as cm
from sklearn import metrics
import cv2

import glob
import pandas as pd
import numpy as np

from util import Utilities

cnn_model = "model/mnist_cnn_tx.h5"
bn_model = "model/mnist_bn_tx.h5"
test_folder = "test_data"
test_label = "test_data/test_data.csv"
batch_size = 100

class DataProcessor:
    def __init__(self):
        print("\n Preparing Dataset...")

    # load train and test dataset
    def load_mnist_dataset(self):
        # load dataset
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        # reshape dataset to have a single channel
        train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
        test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
        # one hot encode target values
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)
        return train_x, train_y, test_x, test_y
        # return train_x[:5000], train_y[:5000], test_x[:500], test_y[:500]

    # scale pixels
    def prep_pixels(self, train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm


    def load_test_images(self, filename):
        # print("File: ", filename)
        img = (255 - cv2.imread(filename, 0))
        image = cv2.resize(img, (28, 28))
        # pyplot.imshow(image, cmap=cm.gray)
        # pyplot.show()
        img = np.array(image)
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

    def data_agumentation(self, train_x, train_y, val_x, val_y):
        # transform train data to be more robust
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                 height_shift_range=0.08, zoom_range=0.08)

        batches = gen.flow(train_x, train_y, batch_size=100)
        val_batches = gen.flow(val_x, val_y, batch_size=100)
        return batches, val_batches


class DigitClassifier:
    def __init__(self):
        print("\n Initializing ML Model...")

    # define cnn model
    def get_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_bn_model(self, mean_px, std_px):
        def standardize(x):
            return (x - mean_px) / std_px
        model = Sequential([
                    Lambda(standardize, input_shape=(28,28,1)),
                    Convolution2D(32,(3,3), activation='relu'),
                    BatchNormalization(axis=1),
                    Convolution2D(32,(3,3), activation='relu'),
                    MaxPooling2D(),
                    BatchNormalization(axis=1),
                    Convolution2D(64,(3,3), activation='relu'),
                    BatchNormalization(axis=1),
                    Convolution2D(64,(3,3), activation='relu'),
                    MaxPooling2D(),
                    Flatten(),
                    BatchNormalization(),
                    Dense(100, activation='relu'),
                    BatchNormalization(),
                    Dense(10, activation='softmax')
                    ])
        model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def train_bn_on_mnist():
    data = DataProcessor()
    # load data
    train_x, train_y, val_x, val_y = data.load_mnist_dataset()

    # prepare pixel data
    train_x, val_x = data.prep_pixels(train_x, val_x)

    mean_px = train_x.mean().astype(np.float32)
    std_px = train_x.std().astype(np.float32)

    # transform train data to be more robust
    data_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.12, shear_range=0.3,
                             height_shift_range=0.12, zoom_range=0.1)

    # call model and train it
    digit_clf = DigitClassifier()
    model = digit_clf.get_bn_model(mean_px, std_px)
    model.optimizer.lr = 0.01
    history = model.fit_generator(data_gen.flow(train_x, train_y, batch_size=batch_size),
                                  epochs=5, validation_data=(val_x, val_y),
                                  verbose=1, steps_per_epoch=train_x.shape[0] // batch_size)

    # history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=5,
    #                                 validation_data=val_batches, validation_steps=val_batches.n)
    model.save(bn_model)

    util = Utilities()
    util.plot_train_result(history)

def train_cnn_on_mnist():
    data = DataProcessor()
    #load data
    train_x, train_y, val_x, val_y = data.load_mnist_dataset()

    #prepare pixel data
    train_x, val_x = data.prep_pixels(train_x, val_x)

    #build the model
    digit_clf = DigitClassifier()
    model = digit_clf.get_cnn_model()

    #transform train data to be more robust
    data_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.12, shear_range=0.3,
                             height_shift_range=0.12, zoom_range=0.1)

    #fit the model with agumented data
    history = model.fit_generator(data_gen.flow(train_x, train_y, batch_size=batch_size),
                                  epochs=15, validation_data=(val_x, val_y),
                                  verbose=1, steps_per_epoch=train_x.shape[0] // batch_size)
    # history = c_model.fit(train_x, train_y, epochs=15, batch_size=64, verbose=1, validation_data=(val_x, val_y))
    model.save(cnn_model)

    util = Utilities()
    util.plot_train_result(history)


def test_on_covid_data(model_name):
    test_y = pd.read_csv(test_label, header=0, index_col=0, squeeze=True).to_dict()
    # print("test: ", test_y)
    ml_model = models.load_model(model_name)

    data = DataProcessor()
    true_y = []
    pred_y = []
    error_file = []
    correct_class = []
    for file in glob.glob(test_folder+"/*.png"):
        test_image = data.load_test_images(file)
        y = ml_model.predict_classes(test_image)
        pred_y.append(y)
        true_y.append(test_y[file.split('/')[1]])
        if y != test_y[file.split('/')[1]]:
            error_file.append([test_image, y, test_y[file.split('/')[1]]])
        else:
            correct_class.append([test_image, y, test_y[file.split('/')[1]]])

    print(metrics.classification_report(true_y, pred_y))
    print("Accuracy : %0.3f" % metrics.accuracy_score(true_y, pred_y))

    confusion_mtx = metrics.confusion_matrix(y_true=true_y, y_pred=pred_y)
    # plot the confusion matrix
    util = Utilities()
    util.plot_confusion_matrix(confusion_mtx, classes=range(10))

    util.display_classification(correct_class)


if __name__ == "__main__":
    # train_cnn_on_mnist()

    # train_bn_on_mnist()

    test_on_covid_data(cnn_model)
