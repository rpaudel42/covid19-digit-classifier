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
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras import models

from matplotlib import pyplot
import matplotlib.cm as cm
from sklearn import metrics
import cv2

import glob
import pandas as pd
import numpy as np

from util import Utilities

digit_model = "model/digit_clf_mnist.h5"

class DataProcessor:
    def __init__(self):
        print("\n Preparing Data...")

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
        # return train_x[:4000], train_y[:4000], test_x[:200], test_y[:200]


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

class CnnClassifier:
    def __init__(self):
        print("\n Initializing training...")

    # define cnn model
    def define_model(self):
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

    # plot learning curves
    def plot_train_result(self, history):
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        pyplot.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['acc'], color='blue', label='train')
        pyplot.plot(history.history['val_acc'], color='orange', label='test')
        pyplot.savefig("results/train_error.pdf")

def train_cnn_on_mnist():

    data = DataProcessor()
    #load data
    train_x, train_y, test_x, test_y = data.load_mnist_dataset()

    #prepare pixel data
    train_x, test_x = data.prep_pixels(train_x, test_x)

    #call model and train it
    cnn_clf = CnnClassifier()
    model = cnn_clf.define_model()
    history = model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1, validation_data=(test_x, test_y))
    cnn_clf.plot_train_result(history)
    model.save(digit_model)


def test_on_covid_data():
    test_folder ="test_data"
    test_y = pd.read_csv(test_folder + "/test_data.csv", header=0, index_col=0, squeeze=True).to_dict()
    # print("test: ", test_y)
    ml_model = models.load_model(digit_model)

    data = DataProcessor()
    true_y = []
    pred_y = []
    error_file = []
    for file in glob.glob(test_folder+"/*.png"):
        test_image = data.load_test_images(file)
        y = ml_model.predict_classes(test_image)
        pred_y.append(y)
        true_y.append(test_y[file.split('/')[1]])
        if y != test_y[file.split('/')[1]]:
            error_file.append([test_image, y, test_y[file.split('/')[1]]])

    print(metrics.classification_report(true_y, pred_y))
    print("Accuracy : %0.3f" % metrics.accuracy_score(true_y, pred_y))

    confusion_mtx = metrics.confusion_matrix(y_true=true_y, y_pred=pred_y)
    # plot the confusion matrix
    util = Utilities()
    util.plot_confusion_matrix(confusion_mtx, classes=range(10))

    util.display_errors(error_file)


if __name__ == "__main__":
    # train_cnn_on_mnist()
    test_on_covid_data()
