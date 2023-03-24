import time
import yaml
import gym
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace

class FgPlanner:
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 100  # PREPROCESS_consecutive_SIZE
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    def __init__(self, robot_scale):
        self.robot_scale = robot_scale
        self.radians_per_elem = None
        self.STRAIGHTS_SPEED = 7.
        self.CORNERS_SPEED = 4.
        self.wall_counters = 0


    def preprocess_lidar(self, ranges):

        proc_ranges = np.array(ranges)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):

        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):

        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE

        # print(averaged_max_gap.argmax())
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2

        return steering_angle

    def wall_detection(self, ranges):
        left_side = ranges[:180]
        right_side = ranges[-180:]

        closest_left = left_side.argmin()
        closest_right = right_side.argmin()

        if np.min(left_side) <= .5 and 30 < closest_left < 150:
            counter = .04 * self.wall_counters
            self.wall_counters += 1

        elif np.min(right_side) <= .5 and 30 < closest_right < 150:
            counter = -.04 * self.wall_counters
            self.wall_counters += 1
        else:
            counter = 0.0
            self.wall_counters = 0

        return counter

    def plan(self, scan_data, odom_data):
        ranges = scan_data

        counter = self.wall_detection(ranges)

        self.radians_per_elem = ((3 / 2) * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges[180:-180])

        # proc_ranges = ranges
        # self.radians_per_elem = ((3 / 2) * np.pi) / len(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = self.find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # best += 10

        steering_angle = self.get_angle(best, len(proc_ranges)) + counter
        # print('steer:', self.get_angle(best, len(proc_ranges)), "added:", counter )
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        # print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        # print(f"Speed: {speed}")


        # x = np.arange(len(ranges[180:-180]))
        # y = np.array(ranges[180:-180])
        # color = ['r' if i == best else 'k' for i in x]
        #
        # plt.scatter(x, y, c=color, s=2)
        # plt.pause(0.01)
        # plt.cla()


        return speed, steering_angle




def plan(pose_x: float, pose_y: float, pose_theta: float, scan_data: list):
    """

    Args:
        pose_x: 자동차의 x 좌표
        pose_y: 자동차의 y 좌표
        pose_theta: 자동차의 heading (theta값)
        scan_data: LiDAR 신호 (1080길이의 배열)

    See Also:
        * LiDAR가 보는 각도는 270도
        * 분해능은 270 / 1080 = 0.25도
        * 따라서 각각의 배열은 0.25도의 차이를 가진 거리값을 가지고 있음

            * scan_data[180]: 자동차 오른쪽
            * scan_data[540]: 자동차 전면
            * scan_data[900]: 자동차 왼쪽

    Returns:
        speed: 원하는 속도 [-20.0 ~ 20.0], steer: 원하는 앵글 값 [0.5 ~ -0.5]
    """


    # 여기에 Wall following 알고리즘 작성

    speed = 1.0
    steer = 0.0

    return speed, steer


def main():
    """
    README:

    * 이 파일을 f1tenth_gym/examples 폴더에 넣어서 실행
    * 건들지 말고 plan 함수만 수정하면 됨
    * ImportError ~~~  numpy 에러가 뜨면, 터미널에서 pip install numpy 실행
    * 데이터 시각화를 하고 싶다면 matplotlib 사용하는것을 권장


    """

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    planner = FgPlanner(0.3302)

    while not done:
        posex = obs['poses_x'][0]
        posey = obs['poses_y'][0]
        poseth = obs['poses_theta'][0]
        scandata = obs['scans'][0]

        # speed, steer = plan(posex, posey, poseth, scandata)
        speed, steer = planner.plan(scandata, [posex, posey, poseth])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()
