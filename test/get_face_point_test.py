import unittest

import numpy as np

from toolkit.read_data import normalize_face_point


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pose_array = np.array([[1, 2, 3, 4]])
        result = normalize_face_point(pose_array)
        print(result)
        self.assertEqual(True, False)
        return True


if __name__ == '__main__':
    unittest.main()
