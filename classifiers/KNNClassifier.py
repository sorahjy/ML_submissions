import time
from tqdm import tqdm

import numpy as np


class KNNClassifier:
    def __init__(self, train_data, train_labels, ord=2):
        """
        初始化KNNClassifier

        :param train_data: 训练集特征
        :type train_data: list
        :param train_labels: 训练集标签
        :type train_labels: list
        :param ord: 正则化项系数
        :type ord: int
        """
        self.ord = ord
        self.train_data = train_data
        self.train_labels = train_labels

    def calculate_distance(self, X, Y):
        dist = np.linalg.norm(X - Y, ord=self.ord)
        return dist

    def find_top_K(self, Y, K):
        """
        寻找最近邻K个值

        :param Y: 测试特征
        :type Y: list
        :param K: KNN中K的值
        :type K: int
        :return: 最近领的K个统计结果
        :rtype: dict
        """
        dist_list = [self.calculate_distance(X, Y) for X in self.train_data]
        sorted_index = np.argsort(dist_list)
        result = {}
        for i in range(K):
            id = sorted_index[i]
            if self.train_labels[id] not in result:
                result[self.train_labels[id]] = 1
            else:
                result[self.train_labels[id]] += 1
        return result

    def predict(self, Y, K=1):
        """
        预测函数

        :param Y: 测试特征
        :type Y: list
        :param K: KNN中K的值
        :type K: int
        :return: 预测结果
        :rtype: Any
        """
        result = self.find_top_K(Y, K)
        return max(result, key=result.get)

    def test_acc(self, test_data, test_label, K=1):
        st = time.clock()
        cnt = 0
        for i in tqdm(range(len(test_data))):
            cnt += 1 if self.predict(test_data[i], K) == test_label[i] else 0
        print("accuracy:", cnt / len(test_data))
        print('time used:', str(time.clock() - st))
