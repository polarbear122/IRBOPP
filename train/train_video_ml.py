# 输入视频流，输出label （0，1）
# 训练图像级别的pose svm分类器,训练时间与数据量的平方成正比，数据量超过一万时很慢。
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
import os
import pickle
import time

import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calculate.calculate as cal
import toolkit.data_resample as data_resample
from log_config import log
from toolkit import read_data
from toolkit.tool import save_model


def sgd_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, loss="log", penalty="l1"))  # 设置训练器
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


def forest_trainer(all_data, all_labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data, all_labels, random_state=1, train_size=0.6, test_size=0.4)
    clf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0, verbose=True, n_jobs=-1)

    x_resampled, y_resampled = data_resample.adasyn(x_train, y_train)
    clf.fit(x_resampled, y_resampled.ravel())  # 对训练集部分进行训练
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
                                      verbose=0, random_state=None, max_iter=2 ** 10))

    # x_train, y_train = data_resample.sample_pipeline(x_train, y_train)
    clf.fit(x_train, y_train.ravel())  # 对训练集部分进行训练
    train_data_score = clf.score(x_train, y_train) * 100
    test_data_score = clf.score(x_test, y_test) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    y_pred = clf.predict(x_test)
    cal.calculate_all(y_test, y_pred)  # 评估计算结果
    s = pickle.dumps(clf)
    return s


def default(_all_data, _all_labels):  # 默认情况下执行的函数
    print('未选择训练器')


if __name__ == "__main__":
    start_at = time.time()
    train_dataset, labels = read_data.read_data_no_track()
    log.logger.info("%s --训练开始--------------" % (os.path.basename(__file__).split(".")[0]))
    get_data_at = time.time()
    name_list = ["SGD", "SVM", "Forest", "LinearSVC"]
    train_model = {"SGD"      : sgd_trainer,
                   "SVM"      : svm_trainer,
                   "Forest"   : forest_trainer,
                   "LinearSVC": linear_svc_trainer,
                   }
    trainer = name_list[0]  # 选择训练器
    log.logger.info(
        "开始训练%s分类器:数据规模(%d,%d),%d" % (trainer, train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))
    model = train_model.get(trainer, default)(train_dataset, labels)  # 执行对应的函数，如果没有就执行默认的函数
    save_model("../trained_model/", trainer + "_video_unsampled_ml.model", model)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("%s:总运行时间%f秒,数据读取时间%f秒,训练时间%f秒" % (trainer, total_con, read_con, train_con))
