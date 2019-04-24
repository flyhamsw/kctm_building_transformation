import numpy as np
import numpy.linalg as la


def solve_ls(A, y):
    N = np.dot(np.transpose(A), A)
    P = np.dot(np.transpose(A), y)
    xsi_hat = np.dot(la.inv(N), P)
    return xsi_hat


def make_A(kctm_building_pairs):
    A = []
    for pair in kctm_building_pairs:
        row_1 = np.array(
            [
                pair['x_kctm'],  # r11
                pair['y_kctm'],  # r12
                pair['z_kctm'],  # r13
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
                pair['x_kctm'],  # r21
                pair['y_kctm'],  # r22
                pair['z_kctm'],  # r23
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
                pair['x_kctm'],  # r31
                pair['y_kctm'],  # r32
                pair['z_kctm'],  # r33
                0,  # tx
                0,  # ty
                1  # tz
            ], dtype=float
        )
        A.append(row_1)
        A.append(row_2)
        A.append(row_3)
    return np.array(A, dtype=float)


def make_y(kctm_building_pairs):
    y = []
    for pair in kctm_building_pairs:
        y.append(pair['x_building'])
        y.append(pair['y_building'])
        y.append(pair['z_building'])
    return np.transpose(np.array(y, dtype=float, ndmin=2))


def rearrange_xsi_hat(xsi_hat):
    r_elements = xsi_hat[0:9]
    T_hat = xsi_hat[9:12]
    R_hat = np.reshape(r_elements, (3, 3))
    # T_hat = np.transpose(np.array(t_elements, ndmin=2))
    return R_hat, T_hat


# kctm_building_pairs = [
#     {
#         'name': '9-0_building',
#         'x_kctm': 199560.818,
#         'y_kctm': 552234.192,
#         'z_kctm': 54.673,
#         'x_building': 0,
#         'y_building': 0,
#         'z_building': 0
#     },
#     {
#         'name': '9-1_building',
#         'x_kctm': 199560.818,
#         'y_kctm': 552234.192,
#         'z_kctm': 54.673 - 0.636,
#         'x_building': 0,
#         'y_building': 0 - 0.636,
#         'z_building': 0
#     },
#     {
#         'name': '10-0_building',
#         'x_kctm': 199561.602,
#         'y_kctm': 552228.201,
#         'z_kctm': 54.715,
#         'x_building': 7.04,
#         'y_building': 0.42,
#         'z_building': 0
#     },
#     {
#         'name': '10-1_building',
#         'x_kctm': 199561.602,
#         'y_kctm': 552228.201,
#         'z_kctm': 54.715 - 0.791,
#         'x_building': 7.04,
#         'y_building': 0.42 - 0.791,
#         'z_building': 0
#     }
# ]

kctm_building_pairs = [
    {
        'name': '9-0_building',
        'x_kctm': 200000.000,
        'y_kctm': 500000.000,
        'z_kctm': 0,
        'x_building': 0,
        'y_building': 0,
        'z_building': 0
    },
    {
        'name': '9-1_building',
        'x_kctm': 200100.000,
        'y_kctm': 500000.000,
        'z_kctm': 0,
        'x_building': 100,
        'y_building': 0,
        'z_building': 0
    },
    {
        'name': '10-0_building',
        'x_kctm': 200000.000,
        'y_kctm': 500100.000,
        'z_kctm': 0,
        'x_building': 0,
        'y_building': 100,
        'z_building': 0
    },
    {
        'name': '10-1_building',
        'x_kctm': 200000.000,
        'y_kctm': 500000.000,
        'z_kctm': 100,
        'x_building': 0,
        'y_building': 0,
        'z_building': 100
    }
]


A = make_A(kctm_building_pairs)

y = make_y(kctm_building_pairs)

xsi_hat = solve_ls(A, y)

R_hat, T_hat = rearrange_xsi_hat(xsi_hat)
print(T_hat)


kctm = np.transpose(
    np.array(
        [
            kctm_building_pairs[0]['x_kctm'],
            kctm_building_pairs[0]['y_kctm'],
            kctm_building_pairs[0]['z_kctm']
        ],
        ndmin=2
    )
)

print(np.dot(R_hat, kctm))
print(np.dot(R_hat, kctm) + T_hat)
