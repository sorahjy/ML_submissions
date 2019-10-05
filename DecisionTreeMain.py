# -*- coding: utf-8 -*-

from math import log
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def data_set_c45_algo(data_set, test_set, feat_name, decrete_label_len):
    entropy = calculate_entropy(data_set)
    data_set_len = len(data_set)
    test_set_len = len(test_set)
    for index in range(decrete_label_len):
        for i in range(data_set_len):
            data_set[i][index] = float(data_set[i][index])
        for i in range(test_set_len):
            test_set[i][index] = float(test_set[i][index])
        allvalue = [vec[index] for vec in data_set]
        sortedallvalue = sorted(allvalue)
        T = []
        for i in range(len(allvalue) - 1):
            T.append(float(sortedallvalue[i] + sortedallvalue[i + 1]) / 2.0)
        best_gain = 0.0
        bestpt = -1.0
        for pt in T:
            nowent = 0.0
            for small in range(2):
                Dt = split_data_for_discrete_index(data_set, index, pt, small)
                p = len(Dt) / float(data_set_len)
                nowent += p * calculate_entropy(Dt)
            if entropy - nowent > best_gain:
                best_gain = entropy - nowent
                bestpt = pt
        feat_name[index] = str(feat_name[index] + "<=" + "%.3f" % bestpt)
        for i in range(data_set_len):
            data_set[i][index] = "yes" if data_set[i][index] <= bestpt else "no"
        for i in range(test_set_len):
            test_set[i][index] = "yes" if test_set[i][index] <= bestpt else "no"

    return data_set, test_set, feat_name


def calculate_entropy(data_set):
    num = len(data_set)
    class_count = {}
    for line in data_set:
        if line[-1] not in class_count:
            class_count[line[-1]] = 0
        class_count[line[-1]] += 1
    entropy = 0.0
    for key, value in class_count.items():
        prob = value / num
        entropy -= prob * log(prob, 2)
    return entropy


def split_data(data_set, axis, value):
    new_data_set = []
    for data in data_set:
        if data[axis] == value:
            tmp_data = data[:axis]
            tmp_data.extend(data[axis + 1:])
            new_data_set.append(tmp_data)
    return new_data_set


def split_data_for_discrete_index(data_set, axis, value, left):
    new_data_set = []
    for featVec in data_set:
        if (left and featVec[axis] <= value) or ((not left) and featVec[axis] > value):
            ret_vec = featVec[:axis]
            ret_vec.extend(featVec[axis + 1:])
            new_data_set.append(ret_vec)
    return new_data_set


def choose_best_feature_index(data_set):
    base_entropy = calculate_entropy(data_set)
    best_index = -1
    best_entropy_inc = 0.0
    for i in range(len(data_set[0]) - 1):
        features = set([exp[i] for exp in data_set])
        total_entropy = 0.0
        for feat in features:
            new_data_set = split_data(data_set, i, feat)
            prob = len(new_data_set) / len(data_set)
            total_entropy += prob * calculate_entropy(new_data_set)
        entropy_inc = (base_entropy - total_entropy) / base_entropy
        if entropy_inc > best_entropy_inc:
            best_entropy_inc = entropy_inc
            best_index = i
    return best_index


def majority_label(class_list):
    class_dict = {}
    for item in class_list:
        if item not in class_dict:
            class_dict[item] = 0
        class_dict[item] += 1
    return max(class_dict, key=class_dict.get)


def build_decision_tree(data_set, labels):
    class_list = [exp[-1] for exp in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_label(class_list)
    index = choose_best_feature_index(data_set)
    if index == -1:
        return majority_label(class_list)
    cur_label = labels[index]
    new_labels = labels[:]
    del (new_labels[index])
    tree = {cur_label: {}}
    unique_feature = set([exp[index] for exp in data_set])
    for feat in unique_feature:
        new_data_set = split_data(data_set, index, feat)
        tree[cur_label][feat] = build_decision_tree(new_data_set, new_labels)
    return tree


def feature_selection(func=DecisionTreeClassifier):
    x, y = load_data()
    clf = func()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
    rfe = RFE(estimator=clf, step=1, n_features_to_select=5)
    rfe = rfe.fit(x_train, y_train)
    print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])
    feat_name = list(x_train.columns[rfe.support_])
    x_train = rfe.transform(x_train)
    x_test = rfe.transform(x_test)
    data_set = np.hstack((x_train, np.array([[i] for i in y_train])))
    test_set = np.hstack((x_test, np.array([[i] for i in y_test])))
    return data_set.tolist(), test_set.tolist(), feat_name


def load_data():
    df = pd.read_csv('datasets/data/brest_cancer_data.csv')
    drop_list = ['id', 'diagnosis', 'Unnamed: 32']
    y = df.diagnosis
    x = df.drop(drop_list, axis=1)
    return x, y


def heatmap_plot():
    x, y = load_data()
    # correlation map
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()


def predict(test_case, tree):
    key = [i for i in tree.keys()][0]
    next = test_case[key]
    if isinstance(tree[key][next], str):
        return tree[key][next]
    else:
        return predict(test_case, tree[key][next])


def test_decision_tree():
    label_dic = {'M': 'malignant', 'B': 'benign'}
    data_set, test_set, feat_name = feature_selection(func=DecisionTreeClassifier)
    data_set_cnv, test_set_cnv, feat_name_cnv = data_set_c45_algo(data_set, test_set, feat_name, len(feat_name))
    decision_tree = build_decision_tree(data_set_cnv, feat_name_cnv)
    print(decision_tree)
    cnt = 0
    for row in test_set_cnv:
        test_case = {'ans': row[-1]}
        for i in range(len(feat_name_cnv)):
            test_case.setdefault(feat_name_cnv[i], row[i])
        ret = predict(test_case, decision_tree)
        if ret == test_case['ans']: cnt += 1
    print('Accuracy:', cnt / len(test_set_cnv))
    return cnt / len(test_set_cnv)


def sklearn_tree_classifier(classfier=DecisionTreeClassifier):
    df = pd.read_csv('datasets/data/brest_cancer_data.csv')
    y = df.diagnosis
    drop_list = ['id', 'diagnosis', 'Unnamed: 32']
    x = df.drop(drop_list, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = classfier(max_depth=4)
    # rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
    rfe = RFE(estimator=clf, step=1, n_features_to_select=5)
    rfe = rfe.fit(x_train, y_train)
    print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])
    feat_name = list(x_train.columns[rfe.support_])
    x_train = x_train[feat_name]
    x_test = x_test[feat_name]
    clf.fit(x_train, y_train)
    ac = accuracy_score(y_test, clf.predict(x_test))
    print('Accuracy is: ', ac)
    le = LabelEncoder()
    for col in x_train:
        x_train[col] = le.fit_transform(x_train[col])
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=x_train.keys(),
                               class_names=clf.classes_,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")
    return ac


def find_best_for_random_forest():
    df = pd.read_csv('datasets/data/brest_cancer_data.csv')
    y = df.diagnosis
    drop_list = ['id', 'diagnosis', 'Unnamed: 32']
    x = df.drop(drop_list, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    param_range = [10, 20, 40, 80, 160, 250]
    train_score, test_score = validation_curve(RandomForestClassifier(), x_train, y_train, param_name='n_estimators',
                                               param_range=param_range, cv=10, scoring='accuracy')
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, train_score, 'o-', color='r', label='training')
    plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('number of tree')
    plt.ylabel('accuracy')
    plt.show()

    train_sizes, train_score, test_score = learning_curve(RandomForestClassifier(), x_train, y_train,
                                                          train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10,
                                                          scoring='accuracy')

    train_error = 1 - np.mean(train_score, axis=1)
    test_error = 1 - np.mean(test_score, axis=1)
    plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('error')
    plt.show()


def sklearn_random_forest():
    df = pd.read_csv('datasets/data/brest_cancer_data.csv')
    y = df.diagnosis
    drop_list = ['id', 'diagnosis', 'Unnamed: 32']
    x = df.drop(drop_list, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=10)
    # rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
    rfe = RFE(estimator=clf, step=1, n_features_to_select=5)
    rfe = rfe.fit(x_train, y_train)
    print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])
    feat_name = list(x_train.columns[rfe.support_])
    x_train = x_train[feat_name]
    x_test = x_test[feat_name]
    clf.fit(x_train, y_train)
    ac = accuracy_score(y_test, clf.predict(x_test))
    print('Accuracy is: ', ac)
    return ac


def evaluate():
    dtm = 0
    dts = 0
    rfs = 0
    for i in tqdm(range(100)):
        dtm += test_decision_tree()
        dts += sklearn_tree_classifier(classfier=DecisionTreeClassifier)
        rfs += sklearn_random_forest()
    return dtm / 100, dts / 100, rfs / 100


if __name__ == "__main__":
    heatmap_plot()
    test_decision_tree()
    sklearn_tree_classifier(classfier=DecisionTreeClassifier)
    find_best_for_random_forest()
    sklearn_random_forest()
    print(evaluate())

    ###### decrepit
    # df = pd.read_csv('datasets/data/brest_cancer_data.csv')
    # drop_list = ['id', 'diagnosis', 'Unnamed: 32']
    # y = df.diagnosis
    # x = df.drop(drop_list, axis=1)
    # data_dia = y
    # data = x
    # data_n_2 = (data - data.mean()) / (data.std())  # standardization
    # data = pd.concat([y, data_n_2.iloc[:, :]], axis=1)
    # data = pd.melt(data, id_vars="diagnosis",
    #                var_name="features",
    #                value_name='value')
    # plt.figure(figsize=(10, 10))
    # sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
    # plt.xticks(rotation=90)
    #
    # sns.jointplot(x.loc[:, 'concavity_worst'], x.loc[:, 'concave points_worst'], kind="regg", color="#ce1414")
    #
    # sns.set(style="white")
    # df = x.loc[:, ['radius_worst', 'perimeter_worst', 'area_worst']]
    # g = sns.PairGrid(df, diag_sharey=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d")
    # g.map_upper(plt.scatter)
    # g.map_diag(sns.kdeplot, lw=3)
    #
    # sns.set(style="whitegrid", palette="muted")
    # data_dia = y
    # data = x
    # data_n_2 = (data - data.mean()) / (data.std())  # standardization
    # data = pd.concat([y, data_n_2.iloc[:, :]], axis=1)
    # data = pd.melt(data, id_vars="diagnosis",
    #                var_name="features",
    #                value_name='value')
    # plt.figure(figsize=(10, 10))
    # tic = time.time()
    # sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    # plt.xticks(rotation=90)
    #
    # # correlation map
    # f, ax = plt.subplots(figsize=(18, 18))
    # sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    # plt.show()
