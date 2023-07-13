from gym_torcs_pi import PIEnv
from sl_agent import PIAgent, DNNAgent
from utils.logger import Logger
from pure_pursuit import PurePursuitModel
import numpy as np
from pathlib import Path


MODEL_TYPE = 'pp'


def run(model_type, race_config_path="raceconfig/agent_practice.xml"):

    assert Path(race_config_path).exists(), f"Path to race_config {race_config_path} does not exist "
    race_config_path = Path(race_config_path).absolute().as_posix()

    assert model_type in ['pi', 'dnn', 'pp'], " Model type should be one of ['pi', 'dnn', 'pp']"

    vision = False
    episode_count = 2
    max_steps = 1500
    step = 0

    kwargs = {'throttle': True, 'brake_change': True, 'gear_change': False}

    env = PIEnv(vision=vision, race_config_path=race_config_path, render=True, **kwargs)

    sl_agent = PIAgent("./training_sl/PI_model") if model_type == "pi" else DNNAgent("./training_sl/DNN_model")

    physics_model = PurePursuitModel()

    logger = Logger(f"Experiments/{model_type}/aalborg")

    print("TORCS Experiment Start.")

    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob, ob_vect = env.reset(relaunch=True, normalize=False)
        else:
            ob, ob_vect = env.reset(normalize=False)

        for j in range(max_steps):

            lookahead = None
            # [steer, accelerate, brake, gear]
            action = {}

            if model_type == 'pp':
                action = physics_model.get_actions(ob)
                lookahead = action['lookahead']

            elif model_type == 'dnn':
                action = sl_agent.get_actions(ob)
                # p_action = physics_model.get_actions(ob)
                # p_action['steer'] = action['steer']
                # action = p_action

            elif model_type == 'pi':
                action = sl_agent.get_actions(ob)
                lookahead = action['lookahead']
                action = physics_model.get_actions(ob, lookahead)

            ob, pre_ob, done = env.step(action, normalize=False)
            print(action)

            logger.store_record(ob, action, lookahead)

            step += 1
            if done:
                break

        logger.save_episode(i)

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":

    run(MODEL_TYPE)
