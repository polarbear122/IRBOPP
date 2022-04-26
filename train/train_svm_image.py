# 训练图像级别的svm分类器
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from toolkit.xml_read import xml_read, str_to_int
from toolkit.read_pose_data import read_json

import numpy as np
import operator
from random import shuffle

from train import svm


def get_svm_train(jaad_anno_path, alpha_pose_path):
    x = []  # key points 0 1 2 3 4 17 18
    y = []  # label looking not-looking
    alpha_pose = read_json(alpha_pose_path)
    jaad_anno = xml_read(jaad_anno_path)
    annotations = jaad_anno["annotations"]
    if "track" not in annotations:
        return np.mat(x), np.mat(y).T
    if "box" in annotations["track"]:
        track_box = [annotations["track"]]
    else:
        track_box = annotations["track"]

    is_repeat = 0
    for i in track_box:
        if i["@label"] != "pedestrian":
            continue
        is_repeat += 1
        if is_repeat >= 2:
            print(jaad_anno_path, "there are two box in one video")
        for annotation in i["box"]:

            xbr = str_to_int(annotation["@xbr"])  # 左上角，右下角 (top-left, bottom-right)
            xtl = str_to_int(annotation["@xtl"])
            ybr = str_to_int(annotation["@ybr"])  # ybr>xbr,ytl>xtl
            ytl = str_to_int(annotation["@ytl"])
            x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            mid_difference = 10000  # 中心点误差
            x_keypoints_proposal = []  # 存储key points
            for pose in alpha_pose:
                if pose["image_id"] == annotation["@frame"] + ".jpg":
                    diff = (pose["box"][0] - x_mid) ** 2 + (pose["box"][1] - y_mid) ** 2
                    if diff < mid_difference:
                        mid_difference = diff
                        x_keypoints_proposal = pose["keypoints"]
            if x_keypoints_proposal:
                x.append(x_keypoints_proposal)
                is_look = annotation["attribute"][2]["#text"]
                if is_look == "looking":
                    y.append(1)
                else:
                    y.append(0)

    print("x.shape,y.shape:", np.mat(x).shape, np.mat(y).T.shape)
    return np.mat(x), np.mat(y).T


if __name__ == "__main__":
    dataSet, labels = np.zeros((1, 78), float), np.zeros((1, 1), float)
    for i in range(1, 20):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        # 1、导入训练数据
        print(i, "------------ 加载数据 --------------")
        x, y = get_svm_train(xml_anno_path, alpha_pose_path)
        if x.shape[1] == 78:
            dataSet = np.concatenate((dataSet, x))
            labels = np.concatenate((labels, y))

    np.savetxt("data.csv", dataSet, delimiter=',')
    np.savetxt("label.csv", labels, delimiter=',')
    # np.savetxt('frame', dataSet, fmt='%f', delimiter=None)
    # 2、训练SVM模型
    print("------------ 训练模型 ---------------")
    C = 1
    toler = 0.001
    maxIter = 5
    svm_model = svm.SVM_training(dataSet, labels, C, toler, maxIter)
    # 3、计算训练的准确性
    print("------------ 计算训练的正确率 --------------")
    accuracy = svm.cal_accuracy(svm_model, dataSet, labels)
    print("训练的正确率是: %.3f%%" % (accuracy * 100))
    # 4、保存最终的SVM模型
    print("------------ 保存模型 ----------------")
    svm.save_svm_model(svm_model, "model_file")
