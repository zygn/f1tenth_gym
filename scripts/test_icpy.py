import numpy as np
import matplotlib.pyplot as plt
from utils import lidar_polar_to_cart, vis_roslidar
import pickle
import os
import cv2
import icp

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

for pkl_name in pkl_list:
    with open(os.path.join(FOLDERPATH, pkl_name), 'rb') as f:
        pkl_dict = pickle.load(f)
        obs_array.append(pkl_dict["obs"])
        action_array.append(pkl_dict["action"])
        frame_names.append(pkl_name)

#2) Visualize before them
if VIS:
    for i, obs in enumerate(obs_array):
        vis_roslidar(obs["scans"], angle_min, angle_inc, idx=i)
        show_rgb(obs["img"])
        cv2.waitKey(10)

# (10, 11) LIKE
# (312, 314) LIKE
# (2730, 2800) LIKE
# (34, 58) UNLIKE

like_pairs = [(10, 11), (312, 314), (2730, 2800)]
#3) Do ICP and visualize output
for pair in like_pairs:
    #a) Convert to cartesian coordinates
    obs0, obs1 = obs_array[10], obs_array[11]
    ranges0, ranges1 = obs0["scans"], obs1["scans"]
    x0, y0 = lidar_polar_to_cart(ranges0, angle_min, angle_inc)
    x1, y1 = lidar_polar_to_cart(ranges1, angle_min, angle_inc)
    scan0 = np.vstack(x0, y0)
    scan1 = np.vstack(x1, y1)

    #b) 