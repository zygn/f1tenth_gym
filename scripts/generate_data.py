import numpy as np
from wrappers import EgoCameraCar
from fgm import FGM
import pickle
import pdb

RENDER = False
FOLDERPATH = './dataset'
env = EgoCameraCar()
agent = FGM(env.angle_min, env.angle_inc)

obs = env.reset()
pdb.set_trace()