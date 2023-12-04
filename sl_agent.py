import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
from pathlib import Path

from agent import Agent
from pure_pursuit import PURE_PURSUIT_2L, RAD_PER_DEG, DEG_PER_RAD, MAX_STEERING_ANGLE_DEG
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DNNAgent(Agent):
    def __init__(self, model_path: str):
        super().__init__('dnn')
        self.model_path = Path(model_path)
        print(self.model_path.joinpath("model.h5").as_posix())
        self._mean = np.load(self.model_path.joinpath("means.npy").as_posix(), allow_pickle=True)
        self._std = np.load(self.model_path.joinpath("stds.npy").as_posix(), allow_pickle=True)
        self._scaler = None
        if self.model_path.joinpath("scaler.pickle").exists():
            with open(self.model_path.joinpath("scaler.pickle").as_posix(), "rb") as fp:
                self._scaler = pickle.load(fp)
                print("Loaded Scaler Pickle")
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path.joinpath("model.h5").as_posix())

    @staticmethod
    def make_observation_big(raw_obs):

        return np.concatenate([[np.array(raw_obs['angle'], dtype=np.float32)],
                               [np.array(raw_obs['distFromStart'], dtype=np.float32)],
                               [np.array(raw_obs['distRaced'], dtype=np.float32)],
                               [np.array(raw_obs['gear'], dtype=np.float32)],
                               np.array(raw_obs['opponents'], dtype=np.float32),
                               [np.array(raw_obs['racePos'], dtype=np.float32)],
                               [np.array(raw_obs['rpm'], dtype=np.float32)],
                               [np.array(raw_obs['speedX'], dtype=np.float32)],
                               [np.array(raw_obs['speedY'], dtype=np.float32)],
                               [np.array(raw_obs['speedZ'], dtype=np.float32)],
                               np.array(raw_obs['track'], dtype=np.float32),
                               [np.array(raw_obs['trackPos'], dtype=np.float32)],
                               np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               [np.array(raw_obs['z'], dtype=np.float32)]
                               ])

    @staticmethod
    def make_observation(raw_obs):

        return np.concatenate([[np.array(raw_obs['angle'], dtype=np.float32)],
                               [np.array(raw_obs['distRaced'], dtype=np.float32)],
                               [np.array(raw_obs['gear'], dtype=np.float32)],
                               np.array(raw_obs['opponents'], dtype=np.float32),
                               [np.array(raw_obs['rpm'], dtype=np.float32)],
                               [np.array(raw_obs['speedX'], dtype=np.float32)],
                               [np.array(raw_obs['speedY'], dtype=np.float32)],
                               [np.array(raw_obs['speedZ'], dtype=np.float32)],
                               np.array(raw_obs['track'], dtype=np.float32),
                               [np.array(raw_obs['trackPos'], dtype=np.float32)],
                               np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               [np.array(raw_obs['z'], dtype=np.float32)]
                               ])

    def get_actions(self, ob):

        state = self.make_observation(ob)

        if self._scaler is not None:
            norm_state = self._scaler.transform(state.reshape((1, -1)))
        else:
            norm_state = ((state - self._mean) / self._std).reshape((1, -1))

        """
        Output sequence:
        'Acceleration', 'Braking', 'Clutch', 'Steering'
        """
        output = self.model.predict(norm_state, batch_size=1).flatten()
        # Formatting output to follow [steer, accelerate, brake, gear]
        # TODO: USE THESE FOR PRODUCTION MODELS: actions = {'accel': output[0], 'brake': output[1], 'clutch': output[2], 'steer': output[3]}
        actions = {'accel': output[0], 'brake': output[1], 'clutch': output[2], 'steer': output[3]}
        # actions = {'steer': output[0]}

        return actions


class PIAgent(Agent):
    def __init__(self, model_path: str):
        super().__init__('pi')
        self.model_path = Path(model_path)
        print(self.model_path.joinpath("model.h5").as_posix())
        self._mean = np.load(self.model_path.joinpath("means.npy").as_posix())
        self._std = np.load(self.model_path.joinpath("stds.npy").as_posix())
        self.model = self.load_model()

    # def load_model(self):
    #
    #     return load_model(self.model_path.joinpath("model.h5").as_posix(), custom_objects={'compute_steering': self.compute_steering,
    #                                                                                        'compute_target_angle': self.compute_target_angle,
    #                                                                                        'get_max_dist_idx': self.get_max_dist_idx})

    def load_model(self):
        return load_model(self.model_path.joinpath("model.h5").as_posix(),
                          custom_objects={'compute_steering': self.compute_steering})


    @staticmethod
    def make_observation(raw_obs):

        return np.concatenate([
                               np.array(raw_obs['opponents'], dtype=np.float32),
                               np.array(raw_obs['track'], dtype=np.float32),
                               [np.array(raw_obs['trackPos'], dtype=np.float32)],
                               [np.array(raw_obs['angle'], dtype=np.float32)],
                               [np.array(raw_obs['rpm'], dtype=np.float32)],
                               [np.array(raw_obs['speedX'], dtype=np.float32)],
            
                               [np.array(raw_obs['speedY'], dtype=np.float32)],
                               [np.array(raw_obs['speedZ'], dtype=np.float32)],
                               np.array(raw_obs['wheelSpinVel'], dtype=np.float32)
                               ])

    def get_actions(self, ob):
        state = self.make_observation(ob)
        norm_state = (state - self._mean) / self._std
        """
        Output sequence:
        'Acceleration', 'Braking', 'Clutch', 'Steering'
        """
        steer, lookahead = self.model.predict(norm_state.reshape(1, -1), batch_size=1)

        # lookahead = np.clip(lookahead, 0, 200)

        actions = {'steer': steer.flatten()[0], 'lookahead': lookahead.flatten()[1]}

        return actions

    @tf.function
    def get_max_dist(self, track_dists):
        return tf.reduce_max(track_dists, axis=1)

    @tf.function
    def get_max_dist_idx(self, track_dists):
        mask = track_dists[:, 9] != 200
        mask = tf.cast(mask, tf.int32)
        indices = tf.math.argmax(track_dists, output_type=tf.int32, axis=1)
        arr1 = indices * mask
        additional = tf.cast(track_dists[:, 9] == 200, tf.int32) * 9
        result = arr1 + additional

        return result

    @tf.function
    def compute_target_angle(self, track_dists):
        max_track_dist_idx = self.get_max_dist_idx(track_dists)

        max_dist_angle = max_track_dist_idx * 10 - 90
        return tf.cast(max_dist_angle, tf.float32) * RAD_PER_DEG

    @tf.function
    def compute_steering(self, data):

        target_angle = data[:, 1] * RAD_PER_DEG
        lookahead = data[:, 0]

        raw_steering_angle_rad = -tf.math.atan(
            (PURE_PURSUIT_2L * tf.math.sin(target_angle)) / (lookahead + 1e-7))

        raw_steering_angle_deg = raw_steering_angle_rad * DEG_PER_RAD

        normalized_steering_angle = tf.clip_by_value(raw_steering_angle_deg / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)
        return normalized_steering_angle

    # @tf.function
    # def compute_steering(self, data):
    #     track, lookahead = data
    #     track = (track * self._std[:19]) + self._mean[:19]
    #     target_angle = self.compute_target_angle(track)  # radians
    #
    #     raw_steering_angle_rad = -tf.math.atan(
    #         PURE_PURSUIT_2L * tf.math.sin(target_angle) / (lookahead + 1e-7))
    #
    #     raw_steering_angle_deg = raw_steering_angle_rad * DEG_PER_RAD
    #
    #     normalized_steering_angle = tf.clip_by_value(raw_steering_angle_deg / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)
    #
    #     return normalized_steering_angle

    @tf.function
    def compute_target_angle_2(self, angle_probs):
        max_track_dist_idx = tf.math.argmax(angle_probs, output_type=tf.int32, axis=1)

        max_dist_angle = max_track_dist_idx * 10 - 90
        return tf.cast(max_dist_angle, tf.float32) * RAD_PER_DEG


    # @tf.function
    # def compute_steering(self, data):
    #     target_angle, lookahead = data
    #
    #     raw_steering_angle_rad = -tf.math.atan(
    #         (PURE_PURSUIT_2L * tf.math.sin(target_angle)) / (lookahead + 1e-7))
    #
    #     raw_steering_angle_deg = raw_steering_angle_rad * DEG_PER_RAD
    #
    #     normalized_steering_angle = tf.clip_by_value(raw_steering_angle_deg / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)
    #     return normalized_steering_angle

    # @staticmethod
    # def get_max_dist(ob):
    #     track_dists = np.array(ob['track'])
    #     return np.max(track_dists)
    #
    # @staticmethod
    # def get_max_dist_idx(ob):
    #     track_dists = np.array(ob['track'])
    #     if track_dists[9] == np.max(track_dists) or track_dists[9] == 200:
    #         return 9
    #     return np.argmax(track_dists)
    #
    # @staticmethod
    # def compute_target_angle(ob):
    #
    #     max_track_dist_idx = PIAgent.get_max_dist_idx(ob)
    #
    #     max_dist_angle = max_track_dist_idx * 10. - 90.
    #     return max_dist_angle * RAD_PER_DEG
    #
    # @staticmethod
    # def make_observation_new(raw_obs):
    #
    #     return np.concatenate([
    #                            np.array(raw_obs['track'], dtype=np.float32),
    #                            [np.array(raw_obs['trackPos'], dtype=np.float32)],
    #                            [np.array(raw_obs['angle'], dtype=np.float32)],
    #                            [np.array(raw_obs['rpm'], dtype=np.float32)],
    #                            [np.array(raw_obs['speedX'], dtype=np.float32)],
    #                            [np.array(raw_obs['speedY'], dtype=np.float32)],
    #                            [np.array(raw_obs['speedZ'], dtype=np.float32)],
    #                            np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
    #                            [np.array(raw_obs['z'], dtype=np.float32)],
    #                            [np.array(PIAgent.compute_target_angle(raw_obs['track']), dtype=np.float32)],
    #                            ])

