import numpy as np
from unrealcv.util import read_png, read_npy
from unrealcv import client
import time

def deg(rad):
    return rad * 180.0/np.pi

class Camera(object):
    """
    My logitech c310 specs:
    60 degree FOV, 1280x720
    """
    def __init__(self, unreal_origin=[1345.0, 3110.0, 132.0+50.]):
        super(Camera, self).__init__()
        self.u_origin = unreal_origin
        client.connect()
    
    def to_upose(self, g_pose):
        """
        g_pose: x, y, theta pose in global cooridnate frame (not unreal)
        1) *100 m->cm 2) flip y axis 3) theta->deg 4)trans to u_origin
        """
        u_pose = [g_pose[0]*100, -1.*g_pose[1]*100, self.u_origin[2], 0., deg(-1.*g_pose[2]), 0.]
        u_pose[0] += self.u_origin[0]
        u_pose[1] += self.u_origin[1]
        return u_pose

    def set_pose(self, u_pose):
        x, y, z, pitch, yaw, roll = u_pose
        client.request("vset /camera/0/pose {} {} {} {} {} {}".format(x, y, z, pitch, yaw, roll))

    def get_img(self):
        """ returns img as rgb npy """
        img = read_npy(client.request('vget /camera/0/lit npy'))
        return img

    def img_at(self, g_pose):
        self.set_pose(self.to_upose(g_pose))
        # time.sleep(0.01)
        return self.get_img()