# 训练图像级别的pose svm分类器,训练时间与数据量的平方成正比，数据量超过一万时很慢。
import pickle
import time

import sklearn
from sklearn import svm

import train.get_data as get_data
from log_config import log


def svm_trainer(train_data, label):
    train_data, test_data, train_label, test_label = \
        sklearn.model_selection.train_test_split(train_data, label, random_state=1, train_size=0.05, test_size=0.95)
    classifier = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')  # 设置训练器
    print("数据分割完成，开始训练svm分类器")
    classifier.fit(train_data, train_label.ravel())  # 对训练集部分进行训练
    train_data_score = classifier.score(train_data, train_label) * 100
    test_data_score = classifier.score(test_data, test_label) * 100
    log.logger.info("训练集正确率:%0.3f%%,测试集正确率:%0.3f%%" % (train_data_score, test_data_score))
    s = pickle.dumps(classifier)
    with open(file="trained_model/svm.model", mode="wb+") as f:
        f.write(s)


if __name__ == "__main__":
    start_at = time.time()
    train_dataset, labels = get_data.read_csv_train_label_data(test=False)
    get_data_at = time.time()
    log.logger.info(
        "svm data to be trained:(%d,%d),%d" % (train_dataset.shape[0], train_dataset.shape[1], labels.shape[0]))
    svm_trainer(train_dataset, labels)
    end_at = time.time()
    total_con, read_con, train_con = end_at - start_at, get_data_at - start_at, end_at - get_data_at
    # print('{0} {1} {0}'.format('hello', 'world'))  # 打乱顺序
    log.logger.info("svm:总运行时间%f秒,数据读取时间%f秒,训练时间%f秒" % (total_con, read_con, train_con))
