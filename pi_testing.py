from gym_torcs_pi import PIEnv
from sl_agent import PIAgent, DNNAgent
from utils.logger import Logger
from pure_pursuit import PurePursuitModel
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

RENDER = True
MODEL_TYPE = 'pi'

test_tracks = [('dirt', 'dirt-1')]# [('dirt', 'dirt-1'), ('dirt', 'dirt-2'),  ('road', 'e-track-2'), ('road', 'spring'), ('road', 'ruudskogen'), ('dirt', 'dirt-3')]


def run(model_type, track=None, race_config_path="raceconfig/agent_practice.xml", render=False):

    assert Path(race_config_path).exists(), f"Path to race_config {race_config_path} does not exist "
    race_config_path = Path(race_config_path).absolute().as_posix()

    map_name = track[1] if track else None
    map_cat = track[0] if track else None

    tree = ET.parse(race_config_path)
    root = tree.getroot()
    root[1][1][0].attrib['val'] = map_name if map_name is not None else root[1][1][0].attrib['val']
    root[1][1][1].attrib['val'] = map_cat if map_cat is not None else root[1][1][1].attrib['val']

    tree.write(race_config_path)

    logger = Logger(f"Experiments_Disc/{model_type}/{map_name}")

    assert model_type in ['pi', 'dnn', 'pp'], " Model type should be one of ['pi', 'dnn', 'pp']"

    vision = False
    episode_count = 3
    max_steps = 2000
    step = 0

    kwargs = {'throttle': True, 'brake_change': True, 'gear_change': False}

    env = PIEnv(vision=vision, race_config_path=race_config_path, render=render, **kwargs)

    sl_agent = PIAgent("training_sl/PI_model_new_3") if model_type == "pi" else DNNAgent("training_sl/DNN_model_BIG")

    physics_model = PurePursuitModel()

    print("TORCS Experiment Start.")

    for i in range(episode_count):
        print("Episode : " + str(i))

        # if np.mod(i, 3) == 0:
        #     # Sometimes you need to relaunch TORCS because of the memory leak error
        #     ob, ob_vect = env.reset(relaunch=True, normalize=False)
        # else:
        ob, ob_vect = env.reset(normalize=False)

        for j in range(max_steps):

            # if j % 2 == 0:
            #     continue

            lookahead = None
            target_angle = None
            # [steer, accelerate, brake, gear]
            action = {}

            if model_type == 'pp':
                action = physics_model.get_actions(ob)
                lookahead = action['lookahead']

            elif model_type == 'dnn':
                action = sl_agent.get_actions(ob)
                p_action = physics_model.get_actions(ob)
                p_action['steer'] = action['steer']
                # p_action = action
                action = p_action

            elif model_type == 'pi':
                action = sl_agent.get_actions(ob)
                lookahead = action['lookahead']
                target_angle = action['target_angle']
                # action = physics_model.get_actions(ob, lookahead)
                p_action = physics_model.get_actions(ob)
                p_action['steer'] = action['steer']

                action = p_action

            ob, pre_ob, done = env.step(action, normalize=False)
            # print(action)

            logger.store_record(ob, action, lookahead, target_angle) if i > 0 else None

            step += 1

            if i == 0:
                done = True

            if done:
                print("DONE")
                break

        logger.save_episode(i) if (i > 0 and track is not None) else None

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":

    # run(MODEL_TYPE, render=True)

    for track in test_tracks:
        print(f"RUNNING FOR MAP: {track}")
        run(MODEL_TYPE, track=track, render=RENDER)
