#!/usr/bin/env python3
### From A2C
from baselines import logger
#Customized OpenAI Baselines functions
from utils.cmd_util import make_torcs_env, torcs_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from scripts.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from gym_torcs_wrpd import TorcsEnv
### From Gym Torcs Wrapped
# from gym_torcs_wrpd import TorcsEnv
from scripts.sample_agent import Agent
import numpy as np
#Reading cmd line args
import sys
# import matplotlib.pyplot as plt

def train( num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    #Torqs Env parameters
    vision, throttle, gear_change = False, False, False
    race_config_path = \
        "/home/z3r0/random/rl/gym_torqs/raceconfig/agent_practice.xml"

    env = VecFrameStack(make_torcs_env( num_env, seed,
        vision=vision, throttle=throttle, gear_change=gear_change,
        race_config_path=race_config_path), 4)
    # env = TorcsEnv( vision=vision, throttle=throttle, gear_change=gear_change)

    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = torcs_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='lstm')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()
    train( num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=1)

if __name__ == '__main__':
    main()

#Torcs Env Tests
def torqs_test():
    #Torcs multiple instance test
    vision = False
    episode_count = 30
    max_steps = 100
    reward = 0
    done = False
    step = 0

    # Generate a Torcs environment
    # Race config expected as first argument
    if len( sys.argv) > 1:
        race_config_path = sys.argv[1]
    else:
        ### TODO: Remove Hardcoded
        race_config_path = \
            "/home/z3r0/random/rl/gym_torqs/raceconfig/agent_practice.xml"
            # "/home/z3r0/random/rl/gym_torqs/raceconfig/agent_practice.xml"

    env = TorcsEnv(vision=vision, throttle=False, race_config_path=race_config_path)

    agent = Agent(1)  # steering only

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        total_reward = 0.
        for j in range(max_steps):
            # action = agent.act(ob, reward, done, vision)
            ob, reward, done, _ = env.step( np.random.choice( [ 0,1,2], p=[1/3, 1/3, 1/3]))
            # print( ob.shape)
            # print( ob)
            total_reward += reward

            step += 1
            if done:
                break

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

# torqs_test()
