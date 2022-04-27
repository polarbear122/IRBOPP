# 训练图像级别的svm分类器
import cv2
import sklearn
from toolkit.xml_read import xml_read, str_to_int
from toolkit.read_pose_data import read_json
import numpy as np
from sklearn import svm
import pickle
from train.get_data import get_train_data


def svm_trainer(train_data, label):
    train_data, test_data, train_label, test_label = \
        sklearn.model_selection.train_test_split(train_data, label, random_state=1, train_size=0.6, test_size=0.4)
    classifier = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')  # 设置训练器
    classifier.fit(train_data, train_label.ravel())  # 对训练集部分进行训练
    train_data_score = classifier.score(train_data, train_label) * 100
    test_data_score = classifier.score(test_data, test_label) * 100
    print("训练集正确率:%0.3f%%" % train_data_score)
    print("测试集正确率:%0.3f%%" % test_data_score)
    s = pickle.dumps(classifier)
    with open(file="trained_model/svm.model", mode="wb+") as f:
        f.write(s)


def load_svm_model(file_path):
    with open(file=file_path, mode="wb+") as trained_model:
        s2 = trained_model.read()
        model = pickle.loads(s2)
    # expected = test_y
    # predicted = model1.predict(test_X)
    return model


if __name__ == "__main__":
    train_data_shape = 7  # 训练的数据的列的大小为7，总训练的数据格式为(number_of_data,7)
    train_dataset, labels = np.zeros((1, train_data_shape), float), np.zeros((1, 1), float)
    for i in range(1, 347):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        x, y = get_train_data(xml_anno_path, alpha_pose_path, video_id)
        if x.shape[1] == train_data_shape:
            train_dataset = np.concatenate((train_dataset, x))
            labels = np.concatenate((labels, y))

    np.savetxt("trained_model/svm_data.csv", train_dataset, delimiter=',')
    np.savetxt("trained_model/svm_label.csv", labels, delimiter=',')
    train_dataset = np.asarray(train_dataset)
    labels = np.asarray(labels)
    svm_trainer(train_dataset, labels)
    cv2.destroyAllWindows()
