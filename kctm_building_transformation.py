import numpy as np
import numpy.linalg as la
import csv
import pickle
from matplotlib import pyplot as plt


def import_data(path, global_shift, apply_global_shift=True):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for idx, row in enumerate(reader):
            if apply_global_shift:
                data.append(np.array([float(row[2]) - global_shift['x'], float(row[1]) - global_shift['y'],  # Donhwamun-ro: 1, 2   Sewoon: 2, 1
                                      float(row[3]) - global_shift['z']]))
            else:
                data.append(np.array([float(row[2]), float(row[1]), float(row[3])]))
    data = np.array(data)
    return data


def fit_plane(data):
    A = np.array([data[:, 1], data[:, 2], np.ones(data[:, 0].shape)]).T
    y = np.array(data[:, 0], ndmin=2).T
    xsi_hat = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, y))
    normal = np.array([1, - xsi_hat[0][0], - xsi_hat[1][0]])
    normal = normal / la.norm(normal)
    return normal


def compute_transformation(input_path, output_path, output_mat_path, global_shift):
    # 데이터 가져오면서 global shift 적용하기

    data = import_data(input_path, global_shift)
    # 주어진 점을 가장 잘 포함하는 평면의 방정식 만들기 (plane fitting)
    n = fit_plane(data)
    # transformation_matrix_yxt = compute_translate_and_rotate_xy(n, global_shift)

    # 서측
    # k = np.cross(n, np.array([0, 0, 1]))
    # k = k / la.norm(k)
    #
    # l = np.cross(k, n)
    # l = l / la.norm(l)

    # 동측
    k = np.cross(n, np.array([0, 0, 1]))
    k = - k / la.norm(k)

    l = np.cross(k, n)
    l = - l / la.norm(l)

    rotation_matrix = np.array([k, l, n])

    data_tranformed = np.dot(rotation_matrix, data.T).T

    with open(output_path, 'w') as f:
        for idx, row in enumerate(data_tranformed):
            f.write('%d %f %f %f\n' % (idx + 1, row[0], row[1], row[2]))

    with open(output_mat_path, 'wb') as f:
        transformation_info = {
            'global_shift': global_shift,
            'rotation_matrix': rotation_matrix
        }
        pickle.dump(transformation_info, f)

    return data_tranformed, rotation_matrix


if __name__ == '__main__':
    # global_shift = {
    #     'x': 199560.039,
    #     'y': 552246.931,
    #     'z': 56.700
    # }
    # data_transformed_west, rotation_matrix_west = compute_transformation(
    #     'data/sewoon_west_excluded.txt',
    #     'data/sewoon_west_excluded_transformed.txt',
    #     'data/sewoon_west_excluded_transform_mat.pickle',
    #     global_shift
    # )
    global_shift = {
        'x': 199598.891,
        'y': 552145.686,
        'z': 57.046
    }
    data_transformed_east, rotation_matrix_east = compute_transformation(
        'data/sewoon_east_excluded.txt',
        'data/sewoon_east_excluded_transformed.txt',
        'data/sewoon_east_excluded_transform_mat.pickle',
        global_shift
    )
    # global_shift = {
    #     'x': 199341.820626,
    #     'y': 552199.462439,
    #     'z': 72.206208
    # }
    # data_transformed_east, rotation_matrix_east = compute_transformation(
    #     'data/donhwamun-ro_chunk_1.txt',
    #     'data/donhwamun-ro_chunk_1_transformed.txt',
    #     'data/donhwamun-ro_chunk_1_transform_mat.pickle',
    #     global_shift
    # )

    # plt.figure()
    # plt.boxplot(data_transformed_east.T[2])
    # plt.show()
