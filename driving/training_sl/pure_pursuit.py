import math
import numpy as np
from simple_pid import PID

PI_HALF = math.pi / 2.0 # 90 deg
PI_FOURTHS = math.pi / 4.0 # 45 deg
RAD_PER_DEG = math.pi / 180.0
DEG_PER_RAD = 1.0 / RAD_PER_DEG

DEFAULT_MIN_SPEED = 50
DEFAULT_MAX_SPEED = 250

GEAR_MAX = 6
RPM_MAX = 9500
ACCEL_MAX = 1.0
ACCEL_DELTA = 0.5 # maximum rate of change in acceleration signal, avoid spinning out

BRAKING_MAX = -0.5   # braking signal <= BRAKING_MAX
BRAKING_DELTA = 0.05 # dampen braking to avoid lockup, max rate of chage in braking
WHEELSPIN_ACCEL_DELTA = 0.025
WHEELSPIN_MAX = 5.0 # greater than this value --> loss of control

PURE_PURSUIT_K = 0.35 # bias - increase to reduces steering sensitivity
PURE_PURSUIT_L = 2.4  # approx vehicle wheelbase
PURE_PURSUIT_2L = 2 * PURE_PURSUIT_L;
MAX_STEERING_ANGLE_DEG = 21 # steering lock
USE_STEERING_FILTER = False #
STEERING_FILTER_SIZE = 5

EDGE_AVOIDANCE_ENABLED = True
EDGE_MAX_TRACK_POS = 0.7 # track edge limit
EDGE_STEERING_INPUT = 0.075 # slightly steer away from edge

STALLED_TIMEOUT = 5  # seconds of no significant movement

LEFT_SIDE = 1
RIGHT_SIDE = -1
MIDDLE = 0
Q1 = 1
Q2 = 2
Q3 = 3
Q4 = 4
OFF_TRACK_TARGET_SPEED = 20

controller_options = {"kp": 0.2, "ki": 0, "kd": 0}

steering_values = []



class PurePursuitModel:

    def __init__(self):

        self.cur_accel = 0.

    @staticmethod
    def get_moving_average(readings):
        avg = np.mean(readings)

        if len(readings) == STEERING_FILTER_SIZE:
            readings = np.delete(readings, 0)
        return avg, readings

    @staticmethod
    def is_off_track(ob):
        return np.abs(ob['trackPos']) > 1.0 and np.abs(ob['angle']) > PI_HALF

    def action(self, ob, lookahead=None):
        speed = math.sqrt(ob['speedX'] ** 2 + ob['speedY'] ** 2)

        if lookahead:
            # print(lookahead)
            divider = lookahead
        else:
            divider = 20 if speed < 90 else (PURE_PURSUIT_K * speed)

        steer =  self.compute_steering(ob, divider)

        accelerate, brake, gear = self.compute_speed(ob)

        return [steer, accelerate, brake, gear]

    def compute_steering(self, ob, lookahead):
        global steering_values

        target_angle = self.compute_target_angle(ob)  # radians

        raw_steering_angle_rad = -math.atan(
            PURE_PURSUIT_2L * math.sin(target_angle) / lookahead)

        raw_steering_angle_deg = raw_steering_angle_rad * DEG_PER_RAD

        normalized_steering_angle = np.clip(raw_steering_angle_deg / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)


        if EDGE_AVOIDANCE_ENABLED and not self.is_off_track(ob):
            edge_steering_correction = 0

            if ob['trackPos'] > EDGE_MAX_TRACK_POS and ob['angle'] < 0.005:  # too far left
                edge_steering_correction = -EDGE_STEERING_INPUT
            elif ob['trackPos'] < -EDGE_MAX_TRACK_POS and ob['angle'] > -0.005:  # too far right
                edge_steering_correction = EDGE_STEERING_INPUT

            normalized_steering_angle += edge_steering_correction

        return normalized_steering_angle

    def get_max_dist(self, ob):
        track_dists = np.array(ob['track'])
        return np.max(track_dists)

    def get_max_dist_idx(self, ob):
        track_dists = np.array(ob['track'])
        if track_dists[9] == np.max(track_dists) or track_dists[9] == 200:
            return 9
        return np.argmax(track_dists)

    def compute_target_angle(self, ob):

        max_track_dist_idx = self.get_max_dist_idx(ob)

        max_dist_angle = max_track_dist_idx * 10. - 90.
        return max_dist_angle * RAD_PER_DEG

    