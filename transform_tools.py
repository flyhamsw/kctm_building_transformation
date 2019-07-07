import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import pickle


def load_transform_parameters(wld_path):
    with open(wld_path, 'r') as f:
        data = f.readlines()
        transform_parameters = {
            'x_gsd': float(data[0][0:-1]),
            'y_gsd': float(data[3][0:-1]),
            'ul_x': float(data[4][0:-1]),
            'ul_y': float(data[5][0:-1])
        }
        return transform_parameters


def load_transform_mat(pickle_path):
    with open(pickle_path, 'rb') as f:
        transform_mat = pickle.load(f)
    return transform_mat


def transform_ics_to_bcs(x_ics, y_ics, transform_parameters):
    x_bcs = x_ics * transform_parameters['x_gsd'] + transform_parameters['ul_x']
    y_bcs = y_ics * transform_parameters['y_gsd'] + transform_parameters['ul_y']
    return x_bcs, y_bcs


def transform_bcs_to_ics(x_bcs, y_bcs, transform_parameters):
    x_ics = (x_bcs - transform_parameters['ul_x']) / transform_parameters['x_gsd']
    y_ics = (y_bcs - transform_parameters['ul_y']) / transform_parameters['y_gsd']
    return x_ics, y_ics


def transform_bcs_to_kctm(x_bcs, y_bcs, transform_mat, z_bcs=0):
    # reverse_rotation = np.dot(np.linalg.inv(transform_mat['rotation_matrix']), np.array([x_bcs, y_bcs, z_bcs]))
    reverse_rotation = np.dot(transform_mat['rotation_matrix'].T, np.array([x_bcs, y_bcs, z_bcs]))
    result = reverse_rotation + np.array(
        [transform_mat['global_shift']['x'],
         transform_mat['global_shift']['y'],
         transform_mat['global_shift']['z']]
    )
    return result[0], result[1], result[2]


def transform_kctm_to_bcs(x_kctm, y_kctm, z_kctm, transform_mat):
    global_shifted_xyz = np.array([
        x_kctm - transform_mat['global_shift']['x'],
        y_kctm - transform_mat['global_shift']['y'],
        z_kctm - transform_mat['global_shift']['z']
    ], ndmin=2).T
    result = np.dot(transform_mat['rotation_matrix'], global_shifted_xyz)
    return result[0][0], result[1][0], result[2][0]


def clip_facade_segment(ul_kctm, lr_kctm, image_path, wld_path, transform_mat_path):
    # Get transform parameters of facade image A
    transform_parameters = load_transform_parameters(wld_path)

    # Convert ROI(KCTM) coordinates to BCS
    transform_mat = load_transform_mat(transform_mat_path)
    ul_bcs = transform_kctm_to_bcs(ul_kctm[0], ul_kctm[1], ul_kctm[2], transform_mat)
    lr_bcs = transform_kctm_to_bcs(lr_kctm[0], lr_kctm[1], lr_kctm[2], transform_mat)

    ul_ics = transform_bcs_to_ics(ul_bcs[0], ul_bcs[1], transform_parameters)
    lr_ics = transform_bcs_to_ics(lr_bcs[0], lr_bcs[1], transform_parameters)

    im_orig = imread(image_path)
    im_clip = im_orig[int(round(ul_ics[1])):int(round(lr_ics[1])), int(round(ul_ics[0])):int(round(lr_ics[0]))]

    plt.imshow(im_clip)
    plt.show()


if __name__ == '__main__':
    # Get transform parameters of facade image A
    transform_parameters_A = load_transform_parameters('data/facade_mapping/sewoon_west_190503_FacadeSegment_1cm.wld')

    # Convert ROI(KCTM) coordinates to BCS
    transform_mat = load_transform_mat('data/gcp/sewoon_west_transform_mat.pickle')
    ul_bcs = transform_kctm_to_bcs(199560.273947,552240.662371,54.502553, transform_mat)
    lr_bcs = transform_kctm_to_bcs(199560.975712,552234.368575,50.680164, transform_mat)

    ul_ics = transform_bcs_to_ics(ul_bcs[0], ul_bcs[1], transform_parameters_A)
    lr_ics = transform_bcs_to_ics(lr_bcs[0], lr_bcs[1], transform_parameters_A)

    im_orig = imread('data/facade_mapping/sewoon_west_190503_FacadeSegment_1cm.png')
    im_clip = im_orig[int(round(ul_ics[1])):int(round(lr_ics[1])), int(round(ul_ics[0])):int(round(lr_ics[0]))]

    plt.imshow(im_clip)

    # Get transform parameters of facade image A
    transform_parameters_A = load_transform_parameters('data/facade_mapping/sewoon_west_190603_FacadeSegment_1cm.wld')

    # Convert ROI(KCTM) coordinates to BCS
    transform_mat = load_transform_mat('data/gcp/sewoon_west_transform_mat.pickle')
    ul_bcs = transform_kctm_to_bcs(199560.273947,552240.662371,54.502553, transform_mat)
    lr_bcs = transform_kctm_to_bcs(199560.975712,552234.368575,50.680164, transform_mat)

    ul_ics = transform_bcs_to_ics(ul_bcs[0], ul_bcs[1], transform_parameters_A)
    lr_ics = transform_bcs_to_ics(lr_bcs[0], lr_bcs[1], transform_parameters_A)

    im_orig = imread('data/facade_mapping/sewoon_west_190603_FacadeSegment_1cm.png')
    im_clip = im_orig[int(round(ul_ics[1])):int(round(lr_ics[1])), int(round(ul_ics[0])):int(round(lr_ics[0]))]

    plt.imshow(im_clip, alpha=0.5)
    plt.show()