"""
Interactive script that converts your unreal map screenshot to an actual map
"""
import cv2
import numpy as np
from scipy import stats
import pdb

#SET DEBUGMODE = True to see cv2 output
DEBUGMODE = True
clicks = []
img = None

def draw_circle(event, x, y, flags, params):
    global clicks
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)
        cv2.circle(img, (x,y), 3,(150, 0,0),-1)
        clicks.append((x, y))

def get2clicks(in_img):
    global clicks
    global img
    clicks = []
    img = in_img
    while True:
        cv2.imshow('UnrealMap', in_img)
        cv2.waitKey(1)
        if len(clicks) >=2:
            break
    x1, y1 = clicks[0]
    x2, y2 = clicks[1]
    
    return x1, x2, y1, y2

#FILL THIS IN EVERYTIME
# origin_world = [1343.0, 3104.0, 131.0]
origin_world = [-2310.0, -50.0, -95.0]

filepath = "../maps/unreal_map.png"
input_map = cv2.imread(filepath)
input_map = input_map[30:, :, :]
bg_rgb = np.array([35, 35, 35])
bg = cv2.inRange(input_map, bg_rgb, bg_rgb)
cv2.namedWindow('UnrealMap')
cv2.setMouseCallback('UnrealMap', draw_circle)

# Get length of line
#TODO: Get line_crop coordinates from mouseclick
x_start, x_end, y_start, y_end = get2clicks(input_map)
line_crop_bw = bg[y_start:y_end, x_start:x_end]
line_crop_rgb = input_map[y_start:y_end, x_start:x_end]
fg_idx = np.nonzero(line_crop_bw == 0.)
y, length = stats.mode(fg_idx[0])
if DEBUGMODE:
    y_idxs = np.where(fg_idx[0] == y)[0]
    start_idx, end_idx = y_idxs[0], y_idxs[-1]
    line_idxs = (fg_idx[0][start_idx:end_idx], fg_idx[1][start_idx:end_idx])
    line_crop_rgb[line_idxs] = np.array([255.0, 0., 0.])
    cv2.imshow("DEBUG line_crop", line_crop_rgb)
    print("Press window to continue")
    cv2.waitKey(0)

# Get start position coordinates (center of red_square)
#TODO: Get map_crop coordinates from mouseclick
print("Map Crop Coordinates")
x_start, x_end, y_start, y_end = get2clicks(input_map)
map_crop_bw = bg[y_start:y_end, x_start:x_end]
map_crop_rgb = input_map[y_start:y_end, x_start:x_end]
start_marker_rgb = np.array([41., 41., 255.])
red_square_idxs = np.where(map_crop_rgb == start_marker_rgb)
ry_start, ry_end = red_square_idxs[0][0], red_square_idxs[0][-1]
rx_start, rx_end = red_square_idxs[1][0], red_square_idxs[1][-1]
center_y = (ry_start + (ry_end - ry_start) / 2.)
center_x = (rx_start + (rx_end - rx_start) / 2.)
if DEBUGMODE:
    map_crop_rgb[int(center_y), int(center_x)] = np.array([255., 255., 255.])
    origin_unreal = [center_y, center_x, 0.0]
    print("Click window to continue")
    cv2.imshow("DEBUG start location", map_crop_rgb)
    cv2.waitKey(0)
origin_x = -(center_x * 1./length[0])
origin_y = (center_y - map_crop_rgb.shape[0]) * 1./length[0]
origin_unreal = [origin_x, origin_y, 0.0]

#Convert map to bitmap representation (use map_crop)
bitmap_map = map_crop_bw
bitmap_map[ry_start:ry_end+1, rx_start:rx_end+1] = 255

cv2.imwrite("../maps/unreal.png", bitmap_map)

#Write other data to file
image = "unreal.png"
resolution = 1./length[0]
origin = origin_unreal
negate = 0
occupied_thresh = 0.65 #doesn't matter
free_thresh = 0.196 #doesn't matter
unreal_origin = origin_world
params = f"image: {image}\nresolution: {resolution}\norigin: {origin}\nnegate: {negate}\noccupied_thresh: {occupied_thresh}\nfree_thresh: {free_thresh}\nunreal_origin: {unreal_origin}"

f = open("../maps/unreal.yaml", "w")
f.write(params)
f.close()