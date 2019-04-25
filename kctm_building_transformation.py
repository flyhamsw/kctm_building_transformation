import numpy as np
import numpy.linalg as la

shift_distance = {
    'x': 199592.0,
    'y': 552207.0,
    'z': 50.0,
    'scale': 100000
}


def solve_ls(A, y):
    N = np.dot(np.transpose(A), A)
    P = np.dot(np.transpose(A), y)
    xsi_hat = np.dot(la.inv(N), P)
    return xsi_hat


def make_A(world_building_pairs):
    A = []
    for pair in world_building_pairs:
        if pair['control']:
            row_1 = np.array(
                [
                    pair['x_world'],  # r11
                    pair['y_world'],  # r12
                    pair['z_world'],  # r13
                    0,  # r21
                    0,  # r22
                    0,  # r23
                    0,  # r31
                    0,  # r32
                    0,  # r33
                    1,  # tx
                    0,  # ty
                    0  # tz
                ], dtype=float
            )
            row_2 = np.array(
                [
                    0,  # r11
                    0,  # r12
                    0,  # r13
                    pair['x_world'],  # r21
                    pair['y_world'],  # r22
                    pair['z_world'],  # r23
                    0,  # r31
                    0,  # r32
                    0,  # r33
                    0,  # tx
                    1,  # ty
                    0  # tz
                ], dtype=float
            )
            row_3 = np.array(
                [
                    0,  # r11
                    0,  # r12
                    0,  # r13
                    0,  # r21
                    0,  # r22
                    0,  # r23
                    pair['x_world'],  # r31
                    pair['y_world'],  # r32
                    pair['z_world'],  # r33
                    0,  # tx
                    0,  # ty
                    1  # tz
                ], dtype=float
            )
            A.append(row_1)
            A.append(row_2)
            A.append(row_3)
    return np.array(A, dtype=float)


def make_y(world_building_pairs):
    y = []
    for pair in world_building_pairs:
        if pair['control']:
            y.append(pair['x_building'])
            y.append(pair['y_building'])
            y.append(pair['z_building'])
    return np.transpose(np.array(y, dtype=float, ndmin=2))


def rearrange_xsi_hat(xsi_hat):
    r_elements = xsi_hat[0:9]
    T_hat = xsi_hat[9:12]
    R_hat = np.reshape(r_elements, (3, 3))
    return R_hat, T_hat


def convert_from_world_to_building(x_building, y_building, z_building, x_world, y_world, z_world):
    world = np.transpose(
        np.array([x_world, y_world, z_world], ndmin=2)
    )
    building_estimation = np.dot(R_hat, world) + T_hat
    estimated_x = building_estimation[0][0]
    estimated_y = building_estimation[1][0]
    estimated_z = building_estimation[2][0]
    rmse_x = estimated_x - x_building
    rmse_y = estimated_y - y_building
    rmse_z = estimated_z - z_building
    rmse = la.norm(
        np.sqrt(np.power(np.array([rmse_x, rmse_y, rmse_z]), 2))
    )
    print('estimated x: %f, difference: %f' % (estimated_x, rmse_x))
    print('estimated y: %f, difference: %f' % (estimated_y, rmse_y))
    print('estimated z: %f, difference: %f' % (estimated_z, rmse_z))
    print('RMSE: %f'% rmse)
    print()
    return rmse, estimated_x, estimated_y, estimated_z


world_building_pairs = [
    {
        'name': 'building_01',
        'x_world': 199592.50,
        'y_world': 552207.92,
        'z_world': 51.06,
        'x_building': 0,
        'y_building': 0,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_02',
        'x_world': 199592.49,
        'y_world': 552207.90,
        'z_world': 54.14,
        'x_building': 0,
        'y_building': 3.08,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_03',
        'x_world': 199592.18,
        'y_world': 552210.61,
        'z_world': 54.14,
        'x_building': 2.73,
        'y_building': 3.08,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_04',
        'x_world': 199592.23,
        'y_world': 552210.63,
        'z_world': 51.08,
        'x_building': 0,
        'y_building': 2.73,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_05',
        'x_world': 199592.34,
        'y_world': 552209.13,
        'z_world': 52.17,
        'x_building': 1.22,
        'y_building': 1.11,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_06',
        'x_world': 199592.28,
        'y_world': 552209.79,
        'z_world': 53.82,
        'x_building': 1.88,
        'y_building': 2.76,
        'z_building': 0,
        'control': True
    },
    {
        'name': 'building_07',
        'x_world': 199592.43,
        'y_world': 552208.52,
        'z_world': 53.43,
        'x_building': 0.62,
        'y_building': 2.38,
        'z_building': 0,
        'control': False
    }
]


A = make_A(world_building_pairs)
y = make_y(world_building_pairs)
xsi_hat = solve_ls(A, y)
R_hat, T_hat = rearrange_xsi_hat(xsi_hat)

print('Rotation matrix:')
print(R_hat)
print('Translation matrix:')
print(T_hat)
print()

for pair in world_building_pairs:
    print(pair['name'])
    convert_from_world_to_building(
        pair['x_building'],
        pair['y_building'],
        pair['z_building'],
        pair['x_world'],
        pair['y_world'],
        pair['z_world']
    )
