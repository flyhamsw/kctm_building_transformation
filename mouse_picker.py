from matplotlib import pyplot as plt
from skimage.io import imread

fig, ax = plt.subplots()
ax.imshow(imread('data/facade_mapping/sewoon_west_190603_FacadeSegment_1cm.png'))


def onclick(event):
    if event.button == 3:
        print('xdata=%f, ydata=%f' % (event.xdata, event.ydata))


cid = fig.canvas.mpl_connect('button_press_event', onclick)
