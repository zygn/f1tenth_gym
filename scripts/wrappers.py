import numpy as np
import gym

class DefaultCar(object):
    def __init__(self, spawn_opp=False, map_path='../maps/unreal.yaml', map_img_ext='.png', exec_dir='../build/', scan_fov=4.7, scan_beams=1080, scan_distance_to_base_link=0.275, csv_path=''):
        self.spawn_opp = spawn_opp
        self.scan_distance_to_base_link = 0.275

        self.map_path = map_path
        self.map_img_ext = map_img_ext
        self.exec_dir = exec_dir
        
        self.angle_min = -scan_fov/2.
        self.angle_max = scan_fov/ 2.
        self.angle_inc = scan_fov / scan_beams

        wheelbase = 0.3302
        mass= 3.74
        l_r = 0.17145
        I_z = 0.04712
        mu = 0.523
        h_cg = 0.074
        cs_f = 4.718
        cs_r = 5.4562

        #init gym backend
        self.racecar_env = gym.make('f110_gym:f110-v0')
        self.racecar_env.init_map(self.map_path, self.map_img_ext, False, False)
        self.racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

        # init opponent agent
        # TODO: init by params.yaml
        # TODO: import csv_path for real, right now ignores other agent
        # TODO: HARCODED NOW - import initial_state from the map.yaml
        # TODO: Assuming ego car has obs_idx [0] (like in gym_bridge)
        self.initial_state = {'x':[0., 2.0 if spawn_opp else 2000.], 'y': [0.0, 0.0], 'theta': [np.pi/2., 0.0]}
        # self.obs, _, self.done, _ = self.racecar_env.reset(self.initial_state)

    def step(self, action):
        """
        ARGS: 
            action: {'ego_idx':0, 'speed':[ego_speed, opp_speed], 'steer':[ego_steer (rads), opp_steer (rads)]}
        """
        self.obs, step_reward, self.done, info = self.racecar_env.step(action)
        self.obs['scans'][0] = list(self.obs['scans'][0])
        self.obs['scans'][1] = list(self.obs['scans'][1])
        return self.obs, step_reward, self.done, info
    
    def reset(self):
        self.obs, _, self.done, _ = self.racecar_env.reset(self.initial_state)
        return self.obs

class DefaultCameraCar(DefaultCar):
    def __init__(self, spawn_opp=False, map_path='../maps/unreal.yaml', map_img_ext='.png', exec_dir='../build/', scan_fov=4.7, scan_beams=1080, scan_distance_to_base_link=0.275, csv_path='', cam_height=20.):
        from sim_camera import Camera
        super().__init__(spawn_opp=False, map_path='../maps/unreal.yaml', map_img_ext='.png', exec_dir='../build/', scan_fov=4.7, scan_beams=1080, scan_distance_to_base_link=0.275, csv_path='')

        # TODO: HARCODED NOW - import unreal_origin from map.yaml
        self.cam = Camera(unreal_origin=[1345., 3110., 132.+cam_height])
    
    def ego_pose(self, obs):
        """ obs dict to [x, y, theta] of ego_car"""
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        return [x, y, theta]

    def step(self, action):
        obs, step_reward, done, info = super().step(action)
        img = self.cam.img_at(self.ego_pose(obs))
        obs['img'] = img
        return obs, step_reward, done, info
    
    def reset(self):
        obs = super().reset()
        img = self.cam.img_at(self.ego_pose(obs))
        obs['img'] = img
        return obs
    
class EgoCameraCar(DefaultCameraCar):
    """
    Formats observations so that only ego-car is returned (cleaner)
    """
    def __init__(self, spawn_opp=False, map_path='../maps/unreal.yaml', map_img_ext='.png', exec_dir='../build/', scan_fov=4.7, scan_beams=1080, scan_distance_to_base_link=0.275, csv_path='', cam_height=20.):
        super().__init__(spawn_opp=False, map_path='../maps/unreal.yaml', map_img_ext='.png', exec_dir='../build/', scan_fov=4.7, scan_beams=1080, scan_distance_to_base_link=0.275, csv_path='', cam_height=cam_height)

    def ego_obs(self, obs):
        ego_obs = {}
        for key in obs:
            if isinstance(obs[key], list) and len(obs[key]) == 2:
                ego_obs[key] = obs[key][0]
            else:
                ego_obs[key] = obs[key]
        return ego_obs

    def step(self, action):
        obs, step_reward, done, info = super().step(action)
        return self.ego_obs(obs), step_reward, done, info
    
    def reset(self):
        obs = super().reset()
        return self.ego_obs(obs)