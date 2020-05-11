import numpy as np
import matplotlib.pyplot as plt
from utils import lidar_polar_to_cart, vis_roslidar
import pickle
import os
import cv2
import icp
import matplotlib.pyplot as plt

VIS = True
FOLDERPATH = '../dataset'

def show_rgb(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("camera", bgr)

obs_array = []
action_array = []
frame_names = []

scan_beams = 1080
scan_fov = 4.7
angle_min = -scan_fov/2.
angle_max = scan_fov/2.
angle_inc = scan_fov/scan_beams

#1) Create obs_array and action_array from dataset (like pairs)
pkl_list = os.listdir(FOLDERPATH)
pkl_list.sort()
# like_pairs = [(10, 31), (312, 301), (2730, 2731)]
like_pairs = [(10, 312), (301, 2730), (312, 2731)]

for pair in like_pairs:
    with open(os.path.join(FOLDERPATH, pkl_list[pair[0]]), 'rb') as f:
        pkl_dict = pickle.load(f)
        obs0 = pkl_dict["obs"]
        act0 = pkl_dict["action"] 

    with open(os.path.join(FOLDERPATH, pkl_list[pair[1]]), 'rb') as f:
        pkl_dict = pickle.load(f)
        obs1 = pkl_dict["obs"]
        act1 = pkl_dict["action"] 

    obs_array.append((obs0, obs1))
    action_array.append((act0, act1))

#2) Visualize before them
if VIS:
    for i, pair in enumerate(obs_array):
        for obs in pair:
            vis_roslidar(obs["scans"], angle_min, angle_inc, idx=i)
            show_rgb(obs["img"])
            cv2.waitKey(0)

# (10, 11) LIKE
# (312, 314) LIKE
# (2734, 2813) LIKE
# (34, 58) UNLIKE

#3) ICP
for pair in obs_array:
    #a) Convert to cartesian coordinates
    obs0, obs1 = pair[0], pair[1]
    ranges0, ranges1 = obs0["scans"], obs1["scans"]
    x0, y0 = lidar_polar_to_cart(ranges0, angle_min, angle_inc)
    x1, y1 = lidar_polar_to_cart(ranges1, angle_min, angle_inc)
    scan0 = np.vstack((x0, y0)).T
    scan1 = np.vstack((x1, y1)).T

    #b) call method 
    T, distances, iterations = icp.icp(scan0, scan1, tolerance=1e-3)
    print(np.mean(distances))

    #c) Transform output and visualize 
    out = np.ones((scan0.shape[0], 3))
    out[:, 0:2] = np.copy(scan1)
    out = np.dot(T, out.T).T

    plt.scatter(x0, y0, c='g')
    plt.scatter(x1, y1, c='r')
    plt.scatter(out[:, 0], out[:, 1], c='b')
    plt.show()