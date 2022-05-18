# 训练图像级别的pose svm分类器,训练时间与数据量的平方成正比，数据量超过一万时很慢。
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.
# SGDClassifier.html#sklearn.linear_model.SGDClassifier
import os
import pickle
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calculate.calculate as cal
import toolkit.data_resample as data_resample
from log_config import log
from toolkit import get_data
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

"""
带有 SGD 训练的线性分类器（SVM、逻辑回归等）。
该估计器使用随机梯度下降 (SGD) 学习实现正则化线性模型：每次估计每个样本的损失梯度，并随着强度递减计划（也称为学习率）不断更新模型。
partial_fitSGD 允许通过该方法进行小批量（在线/核外）学习。为了使用默认学习率计划获得最佳结果，数据应具有零均值和单位方差。
此实现适用于表示为特征的密集或稀疏浮点值数组的数据。它拟合的模型可以用损失参数来控制；默认情况下，它适合线性支持向量机 (SVM)。
正则化器是添加到损失函数的惩罚项，它使用平方欧几里德范数 L2 或绝对范数 L1 或两者的组合（弹性网络）将模型参数缩小到零向量。
如果由于正则化器的原因参数更新超过 0.0 值，则更新被截断为 0.0 以允许学习稀疏模型并实现在线特征选择。
"""


def sgd_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    log.logger.info("image训练开始-------------------------------------------")
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, loss="log", penalty="l1"))  # 设置训练器
    # todo  改变loss函数
    # x_train, y_train = sample_pipeline(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def svm_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(all_data, all_labels, random_state=1, train_size=0.05, test_size=0.95)
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')  # 设置训练器

    # x_train, y_train = data_resample.naive_random_under_sample(x_train, y_train)
    # x_train, y_train = data_resample.smote_sample(x_train, y_train)

    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def forest_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    clf = RandomForestRegressor(n_estimators=10, max_depth=12, random_state=0, min_samples_split=8, min_samples_leaf=20,
                                verbose=True, n_jobs=-1)

    # x_train, y_train = data_resample.adasyn(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    y_p = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            y_p[i] = 0
        else:
            y_p[i] = 1
    cal.calculate_all(y_test, y_p)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def grid_search_cv(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.05)
    # 随机森林去进行预测
    rf = RandomForestClassifier()
    param = {"n_estimators"     : [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30],
             "min_samples_split": [15, 20, 25, 30, 35]}
    # 超参数调优
    clf = GridSearchCV(rf, param_grid=param, cv=2, n_jobs=-1)
    print("超参数调优完毕")
    clf.fit(x_train, y_train.ravel())
    print("随机森林预测的准确率为：", clf.score(x_test, y_test))
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def linear_svc_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    clf = make_pipeline(StandardScaler(),
                        svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.0001, C=1.0,
                                      multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None,
                                      verbose=0, random_state=None, max_iter=2 ** 12))

    # x_train, y_train = data_resample.sample_pipeline(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def logistic_regression(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                           intercept_scaling=1, class_weight=None, random_state=None,
                                           solver="liblinear",
                                           max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1,
                                           l1_ratio=None))

    # x_train, y_train = data_resample.sample_pipeline(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    # np.savetxt("image_logistic_regression_label.csv", np.array(y_pred), delimiter=',')
    s = pickle.dumps(clf)
    return s


def default(_all_data, _all_labels):  # 默认情况下执行的函数
    print('未选择训练器')


def read_data():
    data_path = "./data_by_video/all_single/"
    label_path = "./data_by_video/all_single/"
    stream_pose = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
    stream_label = pd.read_csv(data_path + "label1.csv", header=None, sep=',', encoding='utf-8').values
    for str_id in range(2, 347):
        try:
            pose_arr = pd.read_csv(data_path + "data" + str(str_id) + ".csv", header=None, sep=',',
                                   encoding='utf-8').values
            label_arr = pd.read_csv(label_path + "label" + str(str_id) + ".csv", header=None, sep=',',
                                    encoding='utf-8').values
            print(str_id, "shape:", pose_arr.shape, label_arr.shape)
            stream_pose = np.concatenate((stream_pose, pose_arr), axis=0)
            stream_label = np.concatenate((stream_label, label_arr), axis=0)
        except OSError:
            print("data or label ", str_id, "is not exist")
        else:
            print("data has been load ", str_id)
    stream_pose = stream_pose[:, 2:stream_pose.shape[1] - 1]  # 原始数据包含视频id，图片id，标签，需要去除
    return stream_pose, stream_label


if __name__ == "__main__":
    start_at = time.time()
    train_dataset, labels = read_data()
    get_data_at = time.time()
    name_list = ["SGD", "SVM", "Forest", "LinearSVC", "LogisticRegression", "GridSearchCV"]
    train_model = {"SGD"               : sgd_trainer,
                   "SVM"               : svm_trainer,
                   "Forest"            : forest_trainer,
                   "LinearSVC"         : linear_svc_trainer,
                   "LogisticRegression": logistic_regression,
                   "GridSearchCV"      : grid_search_cv,
                   }
    trainer = name_list[5]  # 选择训练器
    func_name = os.path.basename(__file__).split(".")[0]
    log.logger.info("%s --单帧pose训练开始--------------" % func_name)
    log.logger.info(
        "开始训练%s分类器:数据规模(%d,%d),%d" % (trainer, train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))

    model = train_model.get(trainer, default)(train_dataset, labels)  # 执行对应的函数，如果没有就执行默认的函数
    get_data.save_model("../trained_model/", trainer + "_image_unsampled_ml_coco17.model", model)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s--%s:总运行时间%f秒,数据读取时间%f秒,训练时间%f秒" % (func_name, trainer, total_con, read_con, train_con))
