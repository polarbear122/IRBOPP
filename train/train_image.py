# 训练图像级别的pose svm分类器,训练时间与数据量的平方成正比，数据量超过一万时很慢。
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
import os
import pickle
import time

import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calculate.calculate as cal
import toolkit.data_resample as data_resample
from log_config import log
from toolkit import read_data
from toolkit.tool import save_model

"""
带有 SGD 训练的线性分类器（SVM、逻辑回归等）。
该估计器使用随机梯度下降 (SGD) 学习实现正则化线性模型：每次估计每个样本的损失梯度，并随着强度递减计划（也称为学习率）不断更新模型。
partial_fitSGD 允许通过该方法进行小批量（在线/核外）学习。为了使用默认学习率计划获得最佳结果，数据应具有零均值和单位方差。
此实现适用于表示为特征的密集或稀疏浮点值数组的数据。它拟合的模型可以用损失参数来控制；默认情况下，它适合线性支持向量机 (SVM)。
正则化器是添加到损失函数的惩罚项，它使用平方欧几里德范数 L2 或绝对范数 L1 或两者的组合（弹性网络）将模型参数缩小到零向量。
如果由于正则化器的原因参数更新超过 0.0 值，则更新被截断为 0.0 以允许学习稀疏模型并实现在线特征选择。
"""


def sgd_trainer(x_train, x_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=3000, tol=1e-4, n_jobs=-1, l1_ratio=0.1, loss="hinge",
                                      penalty="l1"))  # 设置训练器
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


def svm_trainer(x_train, x_test, y_train, y_test):
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')  # 设置训练器

    x_train, y_train = data_resample.naive_random_under_sample(x_train, y_train)
    x_train, y_train = data_resample.smote_sample(x_train, y_train)

    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def forest_trainer(x_train, x_test, y_train, y_test):
    # clf = RandomForestRegressor(n_estimators=36, max_depth=128, random_state=0, min_samples_split=8,
    #                             min_samples_leaf=64, verbose=True, n_jobs=-1)
    clf = RandomForestClassifier()

    # log.logger.info("random state:%d" % random_state)
    clf = RandomForestClassifier(n_estimators=32, random_state=0, n_jobs=34)
    # x_train, y_train = data_resample.adasyn(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100  # 随机森林法训练结果存在问题，输出是0-1的浮点数，不是0和1
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def forest_trainer_regressor(x_train, x_test, y_train, y_test):
    # clf = RandomForestRegressor(n_estimators=36, max_depth=128, random_state=0, min_samples_split=8,
    #                             min_samples_leaf=64, verbose=True, n_jobs=-1)
    clf = RandomForestRegressor(n_estimators=256, random_state=0, verbose=True, n_jobs=34)
    for random_state in range(100):
        log.logger.info("random state:%d" % random_state)
        clf = RandomForestRegressor(n_estimators=256, random_state=random_state, verbose=True, n_jobs=34)
        # x_train, y_train = data_resample.adasyn(x_train, y_train)
        clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
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


def linear_svc_trainer(x_train, x_test, y_train, y_test):
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


def logistic_regression(x_train, x_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
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


def gradient_booting(x_train, x_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, criterion="friedman_mse",
                                     max_depth=1, random_state=0)
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


if __name__ == "__main__":
    start_at = time.time()
    train_norm_pose, train_label, train_video_length_list, test_norm_pose, test_label, test_video_length_list \
        = read_data.read_data_track()
    # print("test_norm_pose.shape:", test_norm_pose.shape)
    # np.savetxt("test.csv", np.concatenate((train_label, test_label), axis=0), delimiter=',')
    get_data_at = time.time()
    name_list = ["SGD", "SVM", "Forest", "LinearSVC", "LogisticRegression", "GradientBooting"]
    train_model = {"SGD": sgd_trainer,
                   "SVM": svm_trainer,
                   "Forest": forest_trainer,
                   "LinearSVC": linear_svc_trainer,
                   "LogisticRegression": logistic_regression,
                   "GradientBooting": gradient_booting
                   }
    trainer = name_list[2]  # 选择训练器
    log.logger.info("%s 单帧pose训练开始--------------------------------" % (os.path.basename(__file__).split(".")[0]))
    log.logger.info("开始训练%s分类器:训练集数据规模(%d,%d),%d" %
                    (trainer, train_norm_pose.shape[0], train_norm_pose.shape[1], train_label.shape[0]))

    model = train_model.get(trainer, default)(train_norm_pose, test_norm_pose, train_label,
                                              test_label)  # 执行对应的函数，如果没有就执行默认的函数
    save_model("../train/trained_model/image/", trainer + "_image_ml.model", model)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,训练时间%f秒" % (trainer, total_con, read_con, train_con))
