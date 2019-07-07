from matplotlib import pyplot as plt
from skimage.io import imread
from transform_tools import transform_bcs_to_ics, transform_ics_to_bcs, transform_bcs_to_kctm
from transform_tools import load_transform_parameters, load_transform_mat


def mouse_picker(image_path, transform_parameters, transform_mat):
    print('Press RIGHT button of the mouse to query coordinates.')
    fig, ax = plt.subplots()
    ax.imshow(imread(image_path))

    column_coords_list_bcs = []
    column_coords_list_bcs.append((0, 0))
    column_coords_list_bcs.append((6, 0))
    column_coords_list_bcs.append((12, 0))
    column_coords_list_bcs.append((18, 0))
    column_coords_list_bcs.append((24, 0))

    for column_coords_bcs in column_coords_list_bcs:
        column_coords_ics = transform_bcs_to_ics(column_coords_bcs[0], column_coords_bcs[1], transform_parameters)
        plt.plot(column_coords_ics[0], column_coords_ics[1], '*')

    picked_coords_list = []

    def onclick(event):
        if event.button == 3:
            coords = {}
            coords['x_ics'], coords['y_ics'] = event.xdata, event.ydata
            coords['x_bcs'], coords['y_bcs'] = transform_ics_to_bcs(coords['x_ics'],
                                                                    coords['y_ics'],
                                                                    transform_parameters)
            coords['x_kctm'], coords['y_kctm'], coords['z_kctm'] = transform_bcs_to_kctm(coords['x_bcs'],
                                                                                         coords['y_bcs'],
                                                                                         transform_mat)
            plt.plot(coords['x_ics'], coords['y_ics'], 'o')
            picked_coords_list.append(coords)
            fig.canvas.draw()
            print(coords)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return picked_coords_list


transform_parameters = load_transform_parameters('data/facade_mapping/sewoon_west_190603_FacadeSegment_1cm.wld')
transform_mat = load_transform_mat('data/gcp/sewoon_west_transform_mat.pickle')
picked_coords_list = mouse_picker('data/facade_mapping/sewoon_west_190603_FacadeSegment_1cm.png',
                                  transform_parameters, transform_mat)

with open('result.txt', 'w') as f:
    for coords in picked_coords_list:
        f.write('%f,%f,%f\n' % (coords['x_kctm'], coords['y_kctm'], coords['z_kctm']))
