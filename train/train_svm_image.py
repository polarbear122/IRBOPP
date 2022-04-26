# 训练图像级别的svm分类器
import sklearn
from toolkit.xml_read import xml_read, str_to_int
from toolkit.read_pose_data import read_json
import numpy as np
from sklearn import svm
import pickle


def get_face_keypoints_confi(keypoints):
    # key points 0 1 2 3 4 17 18
    if len(keypoints) != 78:
        return []
    face_confi = []
    for i in [0, 1, 2, 3, 4, 17, 18]:
        face_confi.append(keypoints[i * 3 + 2])
    return face_confi


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
                        x_keypoints_proposal = get_face_keypoints_confi(pose["keypoints"])
            if x_keypoints_proposal:
                x.append(x_keypoints_proposal)
                is_look = annotation["attribute"][2]["#text"]
                if is_look == "looking":
                    y.append(1)
                else:
                    y.append(0)

    print("x.shape,y.shape:", np.mat(x).shape, np.mat(y).T.shape)
    return np.mat(x), np.mat(y).T


def svm_trainer(train_data, label):
    train_data, test_data, train_label, test_label = \
        sklearn.model_selection.train_test_split(train_data, label, random_state=1, train_size=0.6, test_size=0.4)
    classifier = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')  # 设置训练器
    classifier.fit(train_data, train_label.ravel())  # 对训练集部分进行训练
    train_data_score = classifier.score(train_data, train_label)
    test_data_score = classifier.score(test_data, test_label)
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
    for i in range(1, 83):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result/" + video_id + "/alphapose-results.json"
        x, y = get_svm_train(xml_anno_path, alpha_pose_path)
        if x.shape[1] == train_data_shape:
            train_dataset = np.concatenate((train_dataset, x))
            labels = np.concatenate((labels, y))
    train_dataset = np.asarray(train_dataset)
    labels = np.asarray(labels)
    np.savetxt("trained_model/data.csv", train_dataset, delimiter=',')
    np.savetxt("trained_model/label.csv", labels, delimiter=',')
    svm_trainer(train_dataset, labels)
