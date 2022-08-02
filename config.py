csv_data = "D:/CodeResp/IRBOPP/train/halpe26_reid/"
jaad_clip = "E:/CodeResp/pycode/DataSet/JAAD_clips/"
jaad_img = "E:/CodeResp/pycode/DataSet/JAAD_image/"
jaad_anno = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/"
jaad_vehicle = "D:/CodeResp/jaad_data/JAAD/annotations_vehicle/"
alpha_pose = "D:/CodeResp/jaad_data/AlphaReidResultNoFast/"
img_save_patch = "C:/datasetzyf/jaad_patch/video_"
dataset_root = "D:/CodeResp/pytorch-train-nn/dataset/txt_init/lab3070/"
IRBOPP = "D:/CodeResp/IRBOPP/"
cross_csv = "D:/CodeResp/IRBOPP/cross/data/"

face_position = [0, 1, 2, 3, 4]  # 脸部特征点位置
head_position = [0, 1, 2, 3, 4, 17, 18]  # 头部特征点位置
half_top_position = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19]  # 上半身位置

train_data_list = [250, 8, 219, 97, 50, 231, 128, 296, 324, 326, 346, 330, 25, 93, 244, 55, 302, 107, 120, 332, 134,
                   215, 41, 85, 266, 6, 344, 116, 34, 144, 307, 257, 252, 236, 209, 61, 200, 59, 264, 265, 123, 223,
                   216, 1, 80, 132, 18, 108, 141, 56, 275, 162, 213, 67, 285, 174, 315, 113, 168, 277, 321, 225, 77,
                   288, 88, 170, 171, 96, 280, 37, 334, 21, 103, 342, 65, 154, 268, 227, 110, 177, 169, 131, 319, 47,
                   165, 248, 267, 283, 323, 333, 5, 136, 179, 201, 210, 20, 325, 287, 101, 53, 149, 269, 243, 172, 186,
                   45, 153, 271, 337, 240, 300, 182, 82, 38, 161, 155, 259, 207, 335, 297, 44, 43, 112, 233, 263, 94,
                   151, 204, 238, 19, 147, 232, 160, 247, 29, 260, 188, 306, 106, 127, 320, 339, 281, 258, 145, 138,
                   133, 226, 76, 70, 212, 234, 262, 23, 220, 54, 84, 303, 39, 237, 72, 211, 31, 35, 150, 63, 13, 92, 58,
                   290, 230, 239, 224, 331, 105, 4, 274, 246, 189, 12, 305, 90, 114, 241, 16, 9, 329, 316, 314, 98, 51,
                   66, 10, 100, 176, 173, 122, 203, 124, 146, 272, 157, 301, 40, 254, 310, 129, 206, 60, 111, 102, 338,
                   312, 298, 273, 340, 46, 318, 64, 130, 221, 293, 276, 309, 158, 217, 125, 291, 163, 91, 255, 261, 22,
                   69, 81, 181, 299, 289, 187, 256, 205, 142]
test_data_list = [2, 3, 7, 11, 14, 15, 17, 24, 26, 27, 28, 30, 32, 33, 36, 42, 48, 49, 52, 57, 62, 68, 71, 73, 74, 75,
                  78, 79, 83, 86, 87, 89, 95, 99, 104, 109, 115, 117, 118, 119, 121, 126, 135, 137, 139, 140, 143, 148,
                  152, 156, 159, 164, 166, 167, 175, 178, 180, 183, 184, 185, 190, 191, 192, 193, 194, 195, 196, 197,
                  198, 199, 202, 208, 214, 218, 222, 228, 229, 235, 242, 245, 249, 251, 253, 270, 278, 279, 282, 284,
                  286, 292, 294, 295, 304, 308, 311, 313, 317, 322, 327, 328, 336, 341, 343, 345]

all_data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
                 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
                 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
                 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
                 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264,
                 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
                 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304,
                 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
                 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344,
                 345, 346]

# 存在cross情况的视频列表
cross_list = [2, 3, 6, 7, 11, 12, 14, 16, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 35, 37, 38, 39, 41, 42, 44, 45,
              46, 47, 49, 50, 53, 54, 56, 57, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81,
              82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107,
              108, 109, 110, 111, 112, 113, 114, 116, 118, 119, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132,
              133, 134, 135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 149, 150, 151, 152, 154, 155, 156,
              157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
              178, 179, 183, 184, 185, 187, 190, 191, 193, 194, 196, 197, 198, 201, 204, 205, 206, 209, 212, 213, 214,
              215, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
              238, 240, 241, 242, 243, 244, 247, 248, 249, 250, 251, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265,
              266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 277, 278, 279, 280, 281, 283, 285, 286, 287, 290, 291,
              292, 293, 294, 295, 297, 298, 299, 301, 302, 303, 305, 306, 307, 309, 310, 311, 312, 313, 314, 315, 316,
              317, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 331, 332, 333, 334, 335, 336, 338, 339, 340, 341,
              345, 346]

jaad_all_videos_train = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 24, 25, 26, 27, 30, 31, 33, 34, 35, 37, 38,
                         39, 47, 49, 50, 51, 52, 54, 56, 57, 60, 61, 62, 64, 66, 69, 74, 77, 78, 79, 80, 81, 83, 85, 86,
                         88, 91, 94, 95, 98, 108, 109, 111, 112, 114, 119, 120, 121, 122, 126, 129, 130, 131, 132, 133,
                         134, 136, 137, 138, 139, 140, 142, 143, 145, 146, 147, 149, 154, 157, 158, 159, 161, 166, 167,
                         168, 169, 171, 174, 175, 176, 180, 182, 184, 185, 186, 188, 189, 190, 191, 192, 194, 195, 196,
                         198, 200, 202, 204, 205, 207, 208, 209, 210, 214, 215, 218, 219, 220, 225, 227, 228, 229, 231,
                         232, 233, 235, 236, 237, 240, 241, 242, 246, 247, 248, 249, 250, 254, 255, 256, 257, 258, 259,
                         260, 261, 262, 264, 266, 268, 269, 272, 275, 276, 281, 282, 283, 284, 286, 289, 290, 293, 296,
                         297, 298, 301, 302, 310, 311, 312, 315, 317, 318, 319, 320, 321, 323, 324, 325, 326, 328, 331,
                         335, 341, 342, 345, 346]
jaad_all_videos_val = [2, 6, 21, 40, 41, 44, 65, 72, 73, 82, 89, 99, 102, 123, 156, 160, 170, 172, 181, 193, 199, 217,
                       226, 252, 263, 273, 274, 291, 303, 306, 340, 343]
jaad_all_videos_test = [5, 15, 16, 17, 22, 23, 28, 29, 32, 36, 42, 43, 45, 46, 48, 53, 55, 58, 59, 63, 67, 68, 70, 71,
                        75, 76, 84, 87, 90, 92, 93, 96, 97, 100, 101, 103, 104, 105, 106, 107, 110, 113, 115, 116, 117,
                        118, 124, 125, 127, 128, 135, 141, 144, 148, 150, 151, 152, 153, 155, 162, 163, 164, 165, 173,
                        177, 178, 179, 183, 187, 197, 201, 203, 206, 211, 212, 213, 216, 221, 222, 223, 224, 230, 234,
                        238, 239, 243, 244, 245, 251, 253, 265, 267, 270, 271, 277, 278, 279, 280, 285, 287, 288, 292,
                        294, 295, 299, 300, 304, 305, 307, 308, 309, 313, 314, 316, 322, 327, 329, 330, 332, 333, 334,
                        336, 337, 338, 339, 344]
