# 读取xml文件
import logging
import pickle
import cv2
import xmltodict


def xml_read(xml_file_path):
    with open(xml_file_path) as xml_file:
        # xmltodict.parse()方法可以将xml数据转为python中的dict字典数据
        parser_data = xmltodict.parse(xml_file.read())
    xml_file.close()
    return parser_data


# 输入字符类型的浮点数，返回int，用于cv2画图
def str_to_int(float_str):
    return round(float(float_str))


def save_model(file_path, file_name, model):
    with open(file=file_path + file_name, mode="wb") as f:
        f.write(model)


def load_model(file_path):
    with open(file=file_path, mode="rb") as trained_model:
        s2 = trained_model.read()
        model = pickle.loads(s2)
    # expected = test_y
    # predicted = model1.predict(test_X)
    return model


if __name__ == "__main__":
    for i in range(1, 300):
        video_id = "video_" + str(i).zfill(4)
        xml_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
