# ******************************************************************************
# util.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 4/18/20   Paudel     Initial version,
# ******************************************************************************
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.cm as cm

class Utilities:
    def __init__(self):
        pass

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        plt.savefig("results/confusion_matrix.pdf")

    def display_errors(self, error_file):
        """ This function shows 6 images with their predicted and real labels"""
        total_error = len(error_file)
        n = 0
        nrows = 5
        ncols = int(len(error_file)/nrows)
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for row in range(nrows):
            for col in range(ncols):
                # pyplot.imshow(error_file[n][0], cmap=cm.gray)
                # pyplot.show()
                # error = errors_index[n]
                ax[row, col].imshow((error_file[n][0]).reshape((28,28)), cmap=cm.gray)
                ax[row, col].set_title(
                    "Predicted :{} True :{}".format(error_file[n][1], error_file[n][2]), fontsize=8)
                n += 1
        # plt.show()
        plt.savefig("results/error.pdf")