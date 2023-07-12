from gym_torcs_pp import PPEnv
from utils.logger import Logger
from scripts.sample_agent import Agent
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
        ### TODO: Remove Hardcoded
        race_config_path = "/scratch/msz6/Gym-Torcs/raceconfig/agent_practice.xml"

    env = PPEnv(vision=vision, race_config_path=race_config_path, render=True)

    agent = Agent(1)  # steering only

    physics_model = PurePursuitModel()

    logger = Logger("Experiments/PP/g-track-2")

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
            # action = agent.act(ob, reward, done, vision)
            action, lookahead = physics_model.action(ob)
            # print(ob[0], action)
            ob, ob_vect, reward, done, _ = env.step(action, continuous=True, normalize=False) # env.step( np.random.choice( [ 0,1,2], p=[1/3, 1/3, 1/3]))
            # print( ob.shape)
            # print( ob)
            total_reward += reward
            logger.store_record(ob, {"steer": action[0], "accel": action[1]}, lookahead)

            step += 1
            if done:
                break

        logger.save_episode(i)

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    torqs_test()