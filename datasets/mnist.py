from os.path import dirname, join
import numpy as np
import struct

module_path = dirname(__file__)


def _load_images(file_name):
    with open(join(module_path, 'data', file_name), 'rb') as binfile:
        buf = binfile.read()
        magic, num, rows, cols = struct.unpack_from('>IIII', buf, 0)
        bits = num * rows * cols
        images = struct.unpack_from('>' + str(bits) + 'B', buf, struct.calcsize('>IIII'))
        images = np.reshape(images, [num, rows * cols])
    return images


def _load_labels(file_name):
    with open(join(module_path, 'data', file_name), 'rb') as binfile:
        buf = binfile.read()
        magic, num = struct.unpack_from('>II', buf, 0)
        labels = struct.unpack_from('>' + str(num) + 'B', buf, struct.calcsize('>II'))
        labels = np.reshape(labels, [num])
    return labels


def load_mnist():
    """
    读取mnist数据集
    """
    train_images_filename = 'train-images-idx3-ubyte'
    train_labels_filename = 'train-labels-idx1-ubyte'
    test_images_filename = 't10k-images-idx3-ubyte'
    test_labels_filename = 't10k-labels-idx1-ubyte'
    train_images = _load_images(train_images_filename)
    train_labels = _load_labels(train_labels_filename)
    test_images = _load_images(test_images_filename)
    test_labels = _load_labels(test_labels_filename)
    return train_images, train_labels, test_images, test_labels
