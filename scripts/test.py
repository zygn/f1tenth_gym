import numpy as np
import gym

#init gym backend
wheelbase = 0.3302
mass= 3.74
l_r = 0.17145
I_z = 0.04712
mu = 0.523
h_cg = 0.074
cs_f = 4.718
cs_r = 5.4562
# init gym backend
map_path = './maps/unreal_map.yaml'
map_img_ext = '.png'
exec_dir = './build'
racecar_env = gym.make('f110_gym:f110-v0')
racecar_env.init_map(map_path, map_img_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

#init opponent agent
initial_state = {'x':[0., 2000.], 'y': [0.0, 0.0], 'theta': [np.pi/2., 0.0]}
obs, _, done, _ = racecar_env.reset(initial_state)