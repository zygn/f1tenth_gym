import numpy as np
from wrappers import EgoCameraCar
from fgm import FGM
import os
import pickle
import matplotlib.pyplot as plt
import pdb

"""
obs.keys() = (['ego_idx', 'scans', 'poses_x', 'poses_y', 'poses_theta', 'linear_vels_x', 'linear_vels_y', 'ang_vels_z', 'collisions', 'collision_angles', 'lap_times', 'lap_counts', 'img'])
"""

RENDER = False
FOLDERPATH = '../dataset/sim_train'
PREFIX = 'env1'
num_saves = 0

def to_deg(rad):
    return rad * 180/np.pi

def save_data(obs, action):
    global num_saves
    if(num_saves == 0 and not os.path.exists(FOLDERPATH)):
        os.mkdir(FOLDERPATH)
    pkl_dict = {"obs":obs, "action":action}
    filename = f"{FOLDERPATH}/{PREFIX}_sim_{num_saves}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pkl_dict, f)
    num_saves+=1

env = EgoCameraCar()
ego_agent = FGM(env.angle_min, env.angle_inc, speed=4.0)
obs = env.reset()
while True:
    #1) Use FGM to get action and save data
    angle, speed = ego_agent.do_FGM(obs['scans'])
    action = {'ego_idx':0, 'speed':[speed, 0.0], 'steer':[angle, 0.0]}
    saved_action = {'angle': to_deg(angle), 'speed': speed}
    save_data(obs, saved_action)

    #2) Step the env
    obs, rew, _, info = env.step(action)