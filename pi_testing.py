from gym_torcs_sl import SLEnv
from gym_torcs_pp import PPEnv
from sl_agent import PIAgent, DNNAgent
from logger import Logger
from pure_pursuit import PurePursuitModel
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

    # env = SLEnv(vision=vision, render=True, race_config_path=race_config_path, brake_change=True, throttle=True, clutch_change=True, gear_change=False)
    env = PPEnv(vision=vision, race_config_path=race_config_path, render=True)

    sl_agent = PIAgent("/home/cognitia/Desktop/phd/torcs/torcs_SL/PI_model")

    physics_model = PurePursuitModel()

    logger = Logger("PI/aalborg")

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
            steer, lookahead = sl_agent.get_actions(ob)
            print(lookahead)
            action, lookahead = physics_model.action(ob, lookahead)
            # [steer, accelerate, brake, gear]
            print(steer, action[0])
            # ob, done = env.step(action, continuous=True, normalize=False)
            ob, ob_vect, reward, done, _ = env.step(action, continuous=True, normalize=False)
            logger.store_record(ob, {"steer": action[0], "accel": action[1]}, lookahead)

            step += 1
            if done:
                break

        logger.save_episode(i)

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    torqs_test()