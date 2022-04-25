# 训练图像级别的svm分类器
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from toolkit.xml_read import xml_read


def train_svm_image(xml_anno_path, output_data_path, video_id):
    # 得到评论，normal_file为存放正常评论的文件，spam_file为存放垃圾评论的文件
    x = []  # key points 0 1 2 3 4 17 18
    y = []  # label looking not-looking
    xml_anno = xml_read(xml_anno_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 随机划分，训练过程暂时没有使用测试数据

    stopword_file = open("stopword.txt", 'r')  # stopwords.txt是停用词存储所在的文件
    stopword_content = stopword_file.read()
    stopword_list = stopword_content.splitlines()
    stopword_file.close()

    count_vect = CountVectorizer(stop_words=stopword_list, token_pattern=r"(?u)\b\w+\b")
    train_count = count_vect.fit_transform(x_train)

    """
    tf-idf chi特征选择；类似将自然语言转成机器能识别的向量
    """
    tfidf_trainformer = TfidfTransformer()
    train_tfidf = tfidf_trainformer.fit_transform(train_count)
    select = SelectKBest(chi2, k=20000)
    train_tfidf_chi = select.fit_transform(train_tfidf, y_train)

    svc = SVC(kernel='linear')
    svc.fit(train_tfidf, y_train)  # 模型训练
    print("train accurancy:", svc.score(train_tfidf, y_train))  # important 准确值
    train_pre = svc.predict(train_tfidf)  # 预测值（结果内容是识别的具体值）
    print(classification_report(train_pre, y_train))  # 输出分类报告（大概就是准确率、召回率）

    with open('svm.pickle', 'wb') as fw:
        pickle.dump(svc, fw)

    with open('count_vect.pickle', 'wb') as fw:
        pickle.dump(count_vect, fw)

    with open('tfidf_trainformer.pickle', 'wb') as fw:
        pickle.dump(tfidf_trainformer, fw)


if __name__ == "__main__":
    for i in range(1, 300):
        video_id = "video_" + str(i).zfill(4)
        xml_anno_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        train_svm_image(xml_anno_path, output_data_path, video_id)
