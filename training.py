from gym_torcs_pp import PPEnv
from ddpg import *
from sample_agent import Agent
from pure_pursuit import PurePursuitModel
import sys
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
#Torcs Env Tests
def torqs_test():
    #Torcs multiple instance test
    vision = False
    render = False
    episode_count = 1000
    train = True
    start_steps = 512
    load_models = False
    max_steps = 100000000
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
            "/home/cognitia/Desktop/phd/torcs/GymTorcs/gym_torqs/raceconfig/agent_practice.xml"
            # "/home/z3r0/random/rl/gym_torqs/raceconfig/agent_practice.xml"

    env = PPEnv(vision=vision, render=render, race_config_path=race_config_path)

    agent = Agent(1)  # steering only

    physics_model = PurePursuitModel()

    trainer = Trainer()

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    if load_models:
        trainer.load_weights()

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            prev_state, prev_state_vector = env.reset(relaunch=True, normalize=True)
        else:
            prev_statem, prev_state_vector = env.reset(normalize=True)

        episodic_reward = 0

        for j in range(max_steps):

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_vector), 0)

            if step > start_steps:
                lookahead = trainer.policy(tf_prev_state, trainer.ou_noise)[0]
            else:
                lookahead = None

            action, lookahead = physics_model.action(prev_state, lookahead=lookahead)
            # Recieve state and reward from environment.
            state, state_vector, reward, done, info = env.step(action, normalize=True)

            trainer.buffer.record((prev_state_vector, lookahead, reward, state_vector))
            episodic_reward += reward

            if train and step > start_steps:
                np.save("reward.npy", trainer.buffer.reward_buffer)
                np.save("state.npy", trainer.buffer.state_buffer)
                np.save("actions.npy", trainer.buffer.action_buffer)
                np.save("next_state.npy", trainer.buffer.next_state_buffer)
                return 

                trainer.buffer.learn(trainer.actor_model, trainer.critic_model,
                        trainer.target_actor, trainer.target_critic, trainer.gamma, trainer.critic_optimizer,
                                     trainer.actor_optimizer)
                trainer.update_target(trainer.target_actor.variables, trainer.actor_model.variables, trainer.tau)
                trainer.update_target(trainer.target_critic.variables, trainer.critic_model.variables, trainer.tau)


            step += 1
            if done:
                break

            prev_state_vector = state_vector
            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        avg_reward_list.append(avg_reward)

        print("Total Step: " + str(step))
        print("")
        if i % 10 == 0:
            trainer.save_weights()

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    torqs_test()