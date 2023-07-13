from scripts.gym_torcs_sl import SLEnv
from utils.logger import Logger
from sl_agent import DNNAgent
import sys
import numpy as np

#Torcs Env Tests
def torqs_test():
    #Torcs multiple instance test
    vision = False
    episode_count = 2
    max_steps = 300
    reward = 0
    done = False
    step = 0

    # Generate a Torcs environment
    # Race config expected as first argument
    if len( sys.argv) > 1:
        race_config_path = sys.argv[1]
    else:
        race_config_path = "/home/cognitia/Desktop/phd/torcs/GymTorcs/gym_torqs/raceconfig/agent_practice.xml"

    env = SLEnv(vision=vision, render=True, race_config_path=race_config_path, brake_change=True, throttle=True, clutch_change=True, gear_change=False)

    sl_agent = DNNAgent("/home/cognitia/Desktop/phd/torcs/torcs_SL/DNN_model")
    logger = Logger("Experiments/DNN/g-track-2")

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob, ob_vect = env.reset(relaunch=True, normalize=False)
        else:
            ob, ob_vect = env.reset(normalize=False)

        total_reward = 0.
        for j in range(max_steps):
            action = sl_agent.get_actions(ob)
            ob, done = env.step(action, continuous=True, normalize=False)
            logger.store_record(ob, {"steer": action[0], "accel": action[1]}, None)

            step += 1
            if done:
                break
        logger.save_episode(i)

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    torqs_test()