import numpy as np

gcp_world = np.array(
    [
        [199592.938, 552206.707, 54.190, 1],
        [199591.751, 552215.798, 53.906, 1],
        [199591.882, 552210.817, 56.596, 1]
    ]
).T

transformation_mat_global = np.array(
    [
        [1, 0, 0, -199589.950000],
        [0, 1, 0, -552201.800000],
        [0, 0, 1, -50.840000],
        [0, 0, 0, 1]
    ]
)

transformation_mat = np.array(
    [
        [-0.105456, 0.994391, 0.008044, -5.796843],
        [-0.049771, -0.013357, 0.998671, -0.041737],
        [0.993178, 0.104916, 0.050901, -3.340039],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ]
)

gcp_kctm = np.dot(transformation_mat, np.dot(transformation_mat_global, gcp_world))
# gcp_kctm = np.dot(transformation_mat_global, gcp_world)
print(gcp_kctm.T)
