import os.path
import shutil


def get_result():
    path = 'D:/CodeResp/jaad_data/AlphaReidResultNoFast/video_'
    new_path = 'D:/CodeResp/jaad_data/new/AlphaReidResultNoFast/video_'
    for i in range(1, 347):
        i_name = str(i).zfill(4)
        path_name = path + i_name
        new_path_name = new_path + i_name
        file_name = path_name + '/alphapose-results.json'
        new_file_name = new_path_name + '/alphapose-results.json'
        if not os.path.exists(new_path_name):
            os.mkdir(new_path_name)
        print(file_name, new_file_name)
        shutil.copy(file_name, new_file_name)


if __name__ == '__main__':
    get_result()
