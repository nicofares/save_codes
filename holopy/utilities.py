import imageio
from scipy.ndimage import gaussian_filter as gaussian_filter
import numpy as np
# import trackpy as tp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import subprocess
# from wraplorenzmie.pylorenzmie.fitting.Localizer import Localizer
from scipy.io import loadmat


class video_reader(object):
    def __init__(
        self, filename, number=None, background=None, dark_count=0, codecs=None
    ):
        self.filename = filename
        self.codecs = codecs
        self.open_video()
        self.number = number
        self.background = background
        self.dark_count = 0

    def open_video(self):
        if self.codecs == None:
            self.vid = imageio.get_reader(self.filename)
        else:
            self.vid = imageio.get_reader(self.filename, self.codecs)

    def close(self):
        self.vid.close()

    def get_image(self, n):
        """Get the image n of the movie"""
        return np.array(self.vid.get_data(n)[:, :, 1])

    def get_next_image(self):
        return np.array(self.vid.get_next_data()[:, :, 1])

    def get_filtered_image(self, n, sigma):
        """Get the image n of the movie and apply a gaussian filter"""
        return gaussian_filter(self.vid.get_data(n)[:, :, 1], sigma=sigma)

    def get_background(self, n, first_im=None, last_im=None):
        """Compute the background over n images spread out on all the movie"""
        if self.number == None:
            print("needs to compute length of the video.")
            self.get_length()
            print("length of video = {}".format(self.number))
        image = self.get_image(1)
        size = (n + 1, image.shape[0], image.shape[1])
        print(size)
        buf = np.empty(size, dtype=np.uint8)
        if first_im == None: 
            start = 1 
        else:
            start = first_im
        if last_im == None: 
            stop = self.number
        else:
            stop = last_im
        length = stop - start + 1
        get_image = np.arange(start, stop, length // n)

        for n, i in enumerate(tqdm(get_image)):
            image = self.get_image(i)
            buf[n, :, :] = image

        if np.mean(buf[-1, :, :]) == 0:
            buf = buf[:-1, :, :]

        self.background = np.median(buf, axis=0)
        return buf, self.background

    def get_length(self):
        """Read the number of frame of vid, can be long with some format as mp4
        so we don't read it again if we already got it"""
        if self.number == None:
            cmd = (
                r"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "
                + self.filename
            )
            self.number = int(subprocess.check_output(cmd, shell=True)) - 1
            return self.number
        else:
            return self.number


def plot_bounding(image, features):
    """
    Method to plot a squares around detected features.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    for feature in features:
        x, y, w, h = [feature['x_p'], feature['y_p'],*feature["bbox"][1:]]
        test_rect = Rectangle(
            xy=(x - w / 2, y - h / 2),
            width=w,
            height=h,
            fill=False,
            linewidth=3,
            edgecolor="r",
        )
        ax.add_patch(test_rect)
    plt.plot()


def normalize(image, background, dark_count=0):
    "Normalize the image by the background, tacking into account the darkcount."
    return (image - dark_count) / (background - dark_count)


def crop(tocrop, x, y, h):
    return tocrop[y - h // 2 : y + h // 2, x - h // 2 : x + h // 2]


# def center_find(image, tp_opts=None, nfringes=None, maxrange=None):
#     """
#     Using the Localizer method of Pylorenzmie to localize images in
#     holograms.
#     """
#     Loc = Localizer(tp_opts=None, nfringes=None, maxrange=None)
#     prediction = Loc.detect(image)

#     return prediction

def open_xyz_mat(pathname, upward=False, version='1'):
    data = loadmat(pathname, squeeze_me=True)
    if version == '1':
        raw_data = np.zeros((len(data['x']), 3))
        raw_data[:,0] = data['x']
        raw_data[:,1] = data['y']
        raw_data[:,2] = data['z']
    elif version == '0':
        raw_data = data["data"][:, 0:3]
    elif version == '2':
        raw_data = np.zeros((len(data['x']), 7))
        raw_data[:,0] = data['x']
        raw_data[:,1] = data['y']
        raw_data[:,2] = data['z']
        raw_data[:,3] = data['dx']
        raw_data[:,4] = data['dy']
        raw_data[:,5] = data['dz']
        raw_data[:,6] = data['redchi']
    else:
        print("Unknown version. Only values accepted = 'new' or 'old'.")
        print("Try again.")
        version = str(input("Data version? Enter old or new: "))
        raw_data = open_xyz_mat(pathname, upward=upward, version=version)
    if upward:
        raw_data[:,2] = - raw_data[:,2]
    del data
    return raw_data

def remove_end_zeros(data):
    try:
        ind = list(data[:,0]).index(0)
        data = data[:ind,:]
    except ValueError:
        print('No pb. No zero in raw data.')
    return data
