import gym

# making the environment
racecar_env = gym.make('f110_gym:f110-v0')

# Initial state
initial_x = [0.0, 2.0]
initial_y = [0.0, 0.0]
initial_theta = [0.0, 0.0]
lap_time = 0.0

# Resetting the environment
obs, step_reward, done, info = racecar_env.reset({'x': initial_x,
                                                  'y': initial_y,
                                                  'theta': initial_theta})
# Simulation loop
while not done:

    # Your agent here
    ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)

    # Stepping through the environment
    action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
    obs, step_reward, done, info = racecar_env.step(action)

    # Getting the lap time
    lap_time += step_reward
