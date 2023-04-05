import numpy as np
import pandas as pd

if __name__ == '__main__':
    pose_arr = pd.read_csv('save_data/new_pose.csv', header=None, sep=',', encoding='utf-8').values
    pose_arr[:, [0, 1]] = pose_arr[:, [1, 0]]
    np.savetxt('save_data/new_pose1.csv', pose_arr, delimiter=',', fmt='%.3e')
