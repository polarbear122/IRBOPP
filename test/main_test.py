import random
import numpy as np


def generate_random_video_list():
    # randint Return random integer in range [a, b], including both end points.
    # randint函数会返回包含两边终点的一个int值
    __i = random.randint(1, 347)
    print(__i)
    rand_schedule = np.random.permutation(range(346)).tolist()  # 返回一个[0-9]中各个数字的数组

    print(rand_schedule[0:207])
    print(rand_schedule[207:347])


if __name__ == "__main__":
    generate_random_video_list()
