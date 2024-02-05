import os
import pickle
from pathlib import Path


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            os.makedirs(self.path.as_posix())

        self.buffer = []

    def store_record(self, ob, actions, lookahead=None, target_angle=None):
        to_store = {}

        to_store["angle"] = ob["angle"]
        to_store["distRaced"] = ob["distRaced"]
        to_store["trackPos"] = ob["trackPos"]
        to_store["speedX"] = ob["speedX"]
        to_store["speedY"] = ob["speedY"]
        to_store["speedZ"] = ob["speedZ"]

        to_store["steer"] = actions['steer']
        to_store["accel"] = actions['accel']
        to_store["lookahead"] = lookahead
        to_store["target_angle"] = target_angle
        to_store["dist_ahead"] = ob['track'][9]

        to_store["track"] = ob["track"]


        self.buffer.append(to_store)

    def save_episode(self, episode):
        with open(self.path.joinpath(f"episode_{str(episode)}.pickle"), "wb") as fp:
            pickle.dump(self.buffer, fp)
