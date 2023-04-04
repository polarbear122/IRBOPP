# 读取xml文件
import pickle

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


def get_anno_by_frame_id(anno_dict, name_list, frame_id):
    anno_list = []
    for i in name_list:
        anno_list.append(int(anno_dict[i][frame_id]))
    return anno_list


def get_anno_by_list(anno_dict, name_list):
    anno_list = []
    for i in name_list:
        anno_list.append(anno_dict[i])
    return anno_list


# 将原有jaad的str id转换为int id
def change_str_id_to_int(str_id):
    s = str_id.split('_')[-1]
    ped_id = ''.join(filter(str.isdigit, s))
    return int(ped_id)


if __name__ == '__main__':
    print(change_str_id_to_int('0_2_674'))
