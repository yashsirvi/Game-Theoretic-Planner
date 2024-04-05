import numpy as np
import math
from collections import deque
import casadi as ca
import matplotlib.pyplot as plt
# from nav_msgs.msg import Path
import time
import sys

from pyglet.gl import GL_POINTS

class DisparityExtenderStable:
    DISPARITY_DIF = 0.5
    CAR_WIDTH = 2.0

    MAX_SPEED = 5
    # MAX_SPEED = 0.1
    # MIN_STRAIGHT_SPEED = 1
    P_PARAM = 0.075  # MAX_SPEED - STABLE_SPEED
    # STABLE_SPEED = 7 - 2
    STABLE_SPEED = 0.025
    ang_list = deque([0, 0])
    speed_list = deque([0, 0])
    MAX_SPEED_INC = 1
    MAX_ANGLE_CHANGE = 2 * 0.0174533

    LEFT_EXTREME = 900
    RIGHT_EXTREME = 180

    MAX_TURN_ANGLE = 45  # [Put the value in DEGREES]
    ANGULAR_JITTER_FACTOR = 0.0872665  # 5 degree [put value in radians]
    TURN_CLEARANCE = 0.0
    WHEEEL_BASE = 0.33
    COF = 1.14
    ACC_G = 9.81
    MAX_DISTANCE = 6
    MIN_DISTANCE = 0.5

    # added to render waypoints
    drawn_waypoints = []

    def vel_ang_cost(self, ang):
        ONE_DEG = 0.0174533
        THRESH_DEG = 0 * ONE_DEG
        DECAY_FACTOR = 0.1
        ret = 0 if ang <= THRESH_DEG else (1 - np.exp(-ang / DECAY_FACTOR))
        return ret

    def vel_dist_cost(self, dist):
        DECAY_FACTOR = 20
        THRESH_DIST = 50
        ret = (
            0
            if dist >= THRESH_DIST
            else (1 - np.exp((dist - THRESH_DIST) / DECAY_FACTOR))
        )
        return ret

    def process_lidar(self, ranges):
        self.radians_per_point = (2 * np.pi) / len(ranges)

        """PREPROCESSING LIDAR DATA"""
        # Remove the quadrant directly behind us
        eighth = int(len(ranges) / 8)
        proc_ranges = np.array(ranges[eighth:-eighth])
        # Find lidar ranges on back left and back right
        back_right = np.array(ranges[0 : (self.RIGHT_EXTREME)])
        back_left = np.array(ranges[(self.LEFT_EXTREME + 1) :])

        # plot the quadrants directly behind us
        # plot back_right and back_left
        fig, ax = plt.subplots( nrows=1, ncols=3)
        plot_eighth = np.array(ranges[0:eighth]) + np.array(ranges[-eighth:])
        angles = np.linspace(-np.pi/2, np.pi/2, len(plot_eighth))
        x_eighth = plot_eighth * np.cos(angles)
        y_eighth = plot_eighth * np.sin(angles)
        ax[0].plot(x_eighth, y_eighth, 'o')
        ax[0].set_title('Eighth')
        
        br_x = back_right * np.cos(np.linspace(-np.pi/2, 0, len(back_right)))
        br_y = back_right * np.sin(np.linspace(-np.pi/2, 0, len(back_right)))
        ax[1].plot(br_x, br_y, 'o')
        ax[1].set_title('Back Right')

        bl_x = back_left * np.cos(np.linspace(0, np.pi/2, len(back_left)))
        bl_y = back_left * np.sin(np.linspace(0, np.pi/2, len(back_left)))
        ax[2].plot(bl_x, bl_y, 'o')
        ax[2].set_title('Back Left')
        plt.show()
        """DISPARITY CALCULATION"""
        disparities = []
        for i in range(len(proc_ranges) - 1):
            # If the threshold is exceeded, a collision is considered possible
            if abs(proc_ranges[i] - proc_ranges[i + 1]) > self.DISPARITY_DIF:
                min_dis = min(
                    proc_ranges[i], proc_ranges[i + 1]
                )  # 计算可能碰撞最小距离 (Calculate the minimum possible collision distance)
                angle_range = math.ceil(
                    math.degrees(math.atan(self.CAR_WIDTH / 2 / min_dis))
                )  # 根据小车宽度计算不会碰撞的最小角度 (Calculate the minimum angle without collision based on the width of the car)
                angle_range += 20  # add tolerance
                # Laser data range to  be cropped
                side_range = (
                    range(int(i - angle_range + 1), i + 1)
                    if proc_ranges[i + 1] == min_dis
                    else range(i + 1, int(i + 1 + angle_range))
                )
                disparities.append((min_dis, side_range))

        # EXTENDING DISPARITIIES
        for min_dis, side_range in disparities:
            for i in side_range:
                if i >= 0 and i < len(proc_ranges):
                    proc_ranges[i] = min(proc_ranges[i], min_dis)

        max_index = np.argmax(proc_ranges)
        max_value = proc_ranges[max_index]
        """TARGET DISTANCE CALCULATION"""
        target_distances = np.where(proc_ranges >= (max_value - 1))[0]
        driving_distance_index = 0
        if len(target_distances) == 1:
            driving_distance_index = target_distances[0]
        else:
            # mid = int()
            driving_distance_index = target_distances[int(len(target_distances) / 2)]

        """ANGLE CALCULATION"""
        #   CALCULATING ANGLE
        lidar_angle = (max_index - len(proc_ranges) / 2) * self.radians_per_point
        angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))

        #   MODIFY THE ANGLE
        if -self.ANGULAR_JITTER_FACTOR < angle < self.ANGULAR_JITTER_FACTOR:
            angle = 0.0
        angle = np.clip(
            angle, np.radians(-self.MAX_TURN_ANGLE), np.radians(self.MAX_TURN_ANGLE)
        )
        min_left = min(back_left)
        min_right = min(back_right)
        safe_angle = angle
        # SAFETY FOR SHARP TURN
        if min_left <= self.TURN_CLEARANCE and angle > 0.0:
            print("waka-waka")
            safe_angle = 0
        elif min_right <= self.TURN_CLEARANCE and angle < 0.0:
            safe_angle = 0

        # safe_angle = angle

        """ VELOCITY  """
        # velocity= self.MAX_SPEED - self.P_param * abs(angle)
        if abs(safe_angle) < 0.0872665:
            # print("yes")
            velocity = self.MAX_SPEED - self.P_PARAM * self.vel_ang_cost(abs(angle))

        else:
            turning_radius = self.WHEEEL_BASE / (2 * np.sin(safe_angle))
            max_velocity = np.sqrt(self.COF * self.ACC_G * abs(turning_radius))
            velocity = max_velocity
        forward_distance = proc_ranges[driving_distance_index]
        head_distance = proc_ranges[int(len(proc_ranges) / 2)]

        if head_distance > self.MAX_DISTANCE:
            # print("Yay")
            velocity = max(
                velocity - self.P_PARAM * self.vel_dist_cost(abs(head_distance)),
                self.STABLE_SPEED,
            )
        elif forward_distance < self.MIN_DISTANCE:
            print("no")
            velocity = -0.5

        if velocity >= 0:
            # Smoothen out velocity profile
            old_vel, cur_vel = list(self.speed_list)
            # pred_vel = cur_vel # cur_vel + (cur_vel - old_vel)
            # weight = np.exp( -abs(velocity-pred_vel)/5 )
            # velocity = (weight*velocity + (1-weight)*pred_vel)
            if abs(velocity - cur_vel) > self.MAX_SPEED_INC:
                velocity = (
                    cur_vel + (1 - 2 * ((velocity - cur_vel) < 0)) * self.MAX_SPEED_INC
                )
            self.speed_list.popleft()
            self.speed_list.append(velocity)
            # for i in list(self.speed_list):
            #     tv += self.WEIGHT_ALPHA * i
            # velocity = tv

        old_ang, cur_ang = list(self.ang_list)
        if abs(safe_angle - cur_ang) > self.MAX_ANGLE_CHANGE:
            safe_angle = (
                cur_ang + (1 - 2 * ((safe_angle - cur_ang) < 0)) * self.MAX_ANGLE_CHANGE
            )
        self.ang_list.popleft()
        self.ang_list.append(safe_angle)
        # for i in list(self.ang_list):
        #     sa += self.WEIGHT_ALPHA * i
        # safe_angle = sa

        # print("--------------------------VEL-------------------------")
        # print(list(self.speed_list))
        # print("--------------------------ANG-------------------------")
        # print(list(self.ang_list))
        # print()

        return velocity, safe_angle
    

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def render_lidar(self, ranges):
        # plot ranges in matplotlib

        # remove inf values
        ranges = np.array(ranges)
        ranges = ranges[ranges < 100]
        angles = np.linspace(-np.pi/2, np.pi/2, len(ranges))
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        plt.plot(x, y, 'o')
        plt.show()

