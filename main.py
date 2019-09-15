from datasets import load_mnist
from classifiers import KNNClassifier
import random, sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

func_mode_list = ['test_acc', 'run_sample']


def terminate():
    print("USAGE...", "function 1: test KNN accuracy, run the following command", "python main.py test_acc",
          "function 2: show plot sample, run the following command",
          "python main.py run_sample", sep='\r\n')
    sys.exit()


if len(sys.argv) < 2:
    terminate()
else:
    mode = sys.argv[1]
    if mode not in func_mode_list:
        terminate()


def show_plot_sample():
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in tqdm(range(25)):
        id = random.randint(0, len(testX) - 1)
        images = np.reshape(testX[id], [28, 28])
        ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 2, "label:" + str(testY[id]))
        ax.text(0, 4, "predict:" + str(knn.predict(testX[id])))
    plt.show()


if __name__ == '__main__':
    trainX, trainY, testX, testY = load_mnist()
    knn = KNNClassifier(train_data=trainX, train_labels=trainY, ord=2)
    if mode == 'run_sample':
        show_plot_sample()
    else:
        knn.test_acc(test_data=testX, test_label=testY, K=1)
