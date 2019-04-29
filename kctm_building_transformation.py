import numpy as np
import numpy.linalg as la
import csv


def import_data(path, apply_global_shift=True):
    data = []
    global_shift = {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0
    }
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for idx, row in enumerate(reader):
            if idx == 0 and apply_global_shift:
                global_shift['x'] = float(row[2])
                global_shift['y'] = float(row[1])
                global_shift['z'] = float(row[3])
            if apply_global_shift:
                data.append(np.array([float(row[2]) - global_shift['x'], float(row[1]) - global_shift['y'],
                                      float(row[3]) - global_shift['z']]))
            else:
                data.append(np.array([float(row[2]), float(row[1]), float(row[3])]))
    data = np.array(data)
    return data, global_shift


def fit_plane(data):
    A = np.array([data[:, 1], data[:, 2], np.ones(data[:, 0].shape)]).T
    y = np.array(data[:, 0], ndmin=2).T
    xsi_hat = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, y))
    normal = np.array([1, - xsi_hat[0][0], - xsi_hat[1][0]])
    normal = normal / la.norm(normal)
    return normal


def compute_translate_and_rotate_xy(normal, global_shift):
    T = np.identity(4)
    T[0][3] = - global_shift['x']
    T[1][3] = - global_shift['y']
    T[2][3] = - global_shift['z']

    a = normal[0]
    b = normal[1]
    c = normal[2]

    d = la.norm([b, c])
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, c / d, -b / d, 0],
            [0, b / d, c / d, 0],
            [0, 0, 0, 1]
        ]
    )
    Ry = np.array(
        [
            [d, 0, -a, 0],
            [0, 1, 0, 0,],
            [a, 0, d, 0],
            [0, 0, 0, 1]

        ]
    )

    transformation_matrix = la.multi_dot([Ry, Rx, T])
    return transformation_matrix


def rotate_z(transformed_data, transformation_matrix_yxt):
    p1 = transformed_data.T[0]
    p2 = transformed_data.T[1]
    dp = p2 - p1

    theta = np.arccos(dp[0] / la.norm(dp[0:2]))
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    )

    transformation_matrix_zyxt = np.dot(Rz, transformation_matrix_yxt)
    return transformation_matrix_zyxt


if __name__ == '__main__':
    # 주어진 점을 가장 잘 포함하는 평면의 방정식 만들기 (plane fitting)
    data, global_shift = import_data('data/sewoon_east.txt')
    normal = fit_plane(data)
    transformation_matrix_yxt = compute_translate_and_rotate_xy(normal, global_shift)

    # 첫 번째 점을 원점으로 하고 (0, 0, 1)을 법선벡터로 하는 평면으로 좌표변환하기
    data, _ = import_data('data/sewoon_east.txt', False)
    data_homogeneous = np.concatenate([data, np.ones((data.shape[0], 1))], 1).T
    transformed_data = np.dot(transformation_matrix_yxt, data_homogeneous)

    # Z 방향 회전시키기
    transformation_matrix_zyxt = rotate_z(transformed_data, transformation_matrix_yxt)
    transformed_data = np.dot(transformation_matrix_zyxt, data_homogeneous)

    # 구한 회전행렬 저장하기
    with open('data/sewoon_east_transform_mat.txt', 'w') as f:
        for row in transformation_matrix_zyxt:
            f.write('%f %f %f %f\n' % (row[0], row[1], row[2], row[3]))

    # 입력한 점 좌표변환 및 저장하기
    with open('data/sewoon_east_transformed.txt', 'w') as f:
        for idx, row in enumerate(transformed_data.T):
            f.write('%d %f %f %f\n' % (idx, row[0], row[1], row[2]))
