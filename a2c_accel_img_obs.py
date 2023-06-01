import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import xavier_initializer

import gym
import numpy as np

from collections import deque
from collections import namedtuple

import sys, os
import random
from time import time
from datetime import datetime
import pickle

from gym_torcs_wrpd_cont import TorcsEnv

from a2c_agent import A2C_Agent

###### Torqs Env parameters
vision, throttle, gear_change = True, True, False
race_config_path = \
"/home/z3r0/random/rl/gym_torqs/raceconfig/agent_practice.xml"
race_speed = 8.0 # Race speed, mainly for rendered anyway
rendering = True # Display the Torcs rendered stuff or run in console

env = TorcsEnv( vision=vision, throttle=throttle, gear_change=gear_change,
race_config_path=race_config_path, race_speed=race_speed,
rendering=rendering)

# A2C hyperparameters
hidden_1 = 100
hidden_2 = 100
# Accel upgrade
hidden_3 = 50
lr_actor = 1e-3
lr_critic = 1e-2
gamma_ = 0.95
frame = 0
num_episodes = 50000
episode = 0

#### REVIEW:Make it automatic later
input_dim = 22 #env.observation_space.shape[0]

tf.reset_default_graph()

# tensorflow graph
states_ = tf.placeholder(tf.float32, [None, input_dim])
actions_ = tf.placeholder(tf.float32, [None, 1])
# Accel
actions_accel_ = tf.placeholder(tf.float32, [None, 1])
returns_ = tf.placeholder(tf.float32, [None, 1])

""" Removed activation function from mu compuytation """
#activation=tf.nn.relu,

actor_scope = "Actor"
with tf.variable_scope(actor_scope):
    h_1_act = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
        name="h_1_act", use_bias=False)
    h_2_act = layers.dense(inputs=h_1_act, units=hidden_2, kernel_initializer=xavier_initializer(), \
        name="h_2_act", use_bias=False)
    h_3_act = layers.dense(inputs=h_2_act, units=hidden_3, kernel_initializer=xavier_initializer(), \
        name="h_3_act", use_bias=False)

    mu_out = layers.dense(inputs=h_2_act, units=1, kernel_initializer=xavier_initializer(), \
        activation=tf.nn.tanh, name="mu_out", use_bias=False)
    sigma_out = layers.dense(inputs=h_2_act, units=1, kernel_initializer=xavier_initializer(), \
        activation=tf.nn.softplus, name="sigma_out", use_bias=False)

    # Accel
    mu_out_accel = layers.dense(inputs=h_3_act, units=1, kernel_initializer=xavier_initializer(), \
        activation=tf.nn.tanh, name="mu_out_accel", use_bias=False)
    sigma_out_accel = layers.dense(inputs=h_3_act, units=1, kernel_initializer=xavier_initializer(), \
        activation=tf.nn.softplus, name="sigma_out_accel", use_bias=False)

    normal_dist = tf.distributions.Normal(loc=mu_out, scale=sigma_out)
    act_out = tf.reshape(normal_dist.sample(1), shape=[-1,1])
    # act_out = tf.clip_by_value(act_out, env.action_space.low, env.action_space.high)
    act_out = tf.clip_by_value(act_out, -1.0, 1.0)

    normal_dist_accel = tf.distributions.Normal(loc=mu_out_accel, scale=sigma_out_accel)
    act_out_accel = tf.reshape(normal_dist_accel.sample(1), shape=[-1,1])
    # act_out_accel = tf.clip_by_value(act_out_accel, env.action_space.low, env.action_space.high)
    act_out_accel = tf.clip_by_value(act_out_accel, -1.0, 1.0)

critic_scope = "Critic"
with tf.variable_scope(critic_scope):
    h_1_cri = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
        name="h_1_cri", use_bias=False)
    h_2_cri = layers.dense(inputs=h_1_cri, units=hidden_2, kernel_initializer=xavier_initializer(), \
        name="h_2_cri", use_bias=False)
    v_out = layers.dense(inputs=h_2_cri, units=1, activation=None, kernel_initializer=xavier_initializer(), \
        name="v_out", use_bias=False)

    #Accel
    h_3_cri = layers.dense(inputs=h_2_cri, units=hidden_3, kernel_initializer=xavier_initializer(), \
        name="h_3_cri", use_bias=False)
    v_out_accel = layers.dense(inputs=h_3_cri, units=1, activation=None, kernel_initializer=xavier_initializer(), \
        name="v_out_accel", use_bias=False)

logprobs = normal_dist.log_prob(actions_)
logprobs_accel = normal_dist_accel.log_prob(actions_accel_)

# for more experiences, add entropy term to loss
entropy = normal_dist.entropy()
advantages = returns_ - v_out_accel
# Accel
entropy_accel = normal_dist_accel.entropy()
advantages_accel = returns_ - v_out_accel


# Define Policy Loss and Value loss
policy_loss = tf.reduce_mean(-logprobs * advantages - 0.01*entropy)
value_loss = tf.reduce_mean(tf.square(returns_ - v_out))

optimizer_policy = tf.train.AdamOptimizer(learning_rate=lr_actor)
optimizer_value = tf.train.AdamOptimizer(learning_rate=lr_critic)

train_policy = optimizer_policy.minimize(policy_loss, \
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Actor"))
train_value = optimizer_value.minimize(value_loss, \
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Critic"))

# Accel
policy_loss_accel = tf.reduce_mean(-logprobs_accel * advantages_accel - 0.01*entropy_accel)
value_loss_accel = tf.reduce_mean(tf.square(returns_ - v_out_accel))

optimizer_policy_accel = tf.train.AdamOptimizer(learning_rate=lr_actor)
optimizer_value_accel = tf.train.AdamOptimizer(learning_rate=lr_critic)

train_policy_accel = optimizer_policy_accel.minimize(policy_loss_accel, \
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Actor"))
train_value_accel = optimizer_value.minimize(value_loss_accel, \
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Critic"))

# Configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# session
sess = tf.Session(config=config)

# initializing all variables
init = tf.global_variables_initializer()
sess.run(init)

#Model saving parameter
save_base_path = os.getcwd() + "/trained_models/accel/"
save_every_how_many_ep = 1

#Stat save
saving_stats = True
stats_base_path = os.getcwd() + "/trained_models/accel/"

#Model loading / restoring
restore_model = False
# restore_base_path = "/tmp/torcs_save/"
restore_base_path = os.getcwd() + "/trained_models/accel"
# restore_file_name = "torcs_a2c_cont_steer_2018-05-20 22:50:16.601@ep_99_scored_64984.tfckpt"
restore_file_name = "fgile.tfckpt"

restore_full_name = restore_base_path + restore_file_name

if restore_model:
    saver = tf.train.Saver()
    saver.restore(sess, restore_full_name)
    print( "##### DEBUG: Restored from: %s" % restore_full_name)

# Stats for plotting
ep_scores = []

# A2C algorithm
for i_ep in range(num_episodes):
    #DEBUG
    print( "Episode %d / %d" % ( i_ep+1, num_episodes) )

    states_list = []
    actions_list = []
    rewards_list = []
    mu_list = []
    sigma_list = []

    # Accel implement
    mu_list_accel = []
    sigma_list_accel = []
    actions_list_accel = []

    ep_reward = 0
    ep_rewards = []

    #Torcs Memory Leak workaround
    # curr_frame = env.reset() #Insted of this, do this =>
    if np.mod(i_ep, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        curr_frame = env.reset(relaunch=True)
    else:
        curr_frame = env.reset()

    done = False
    curr_time = time()
    step = 0

    while not done:
        curr_state = curr_frame.reshape(1, -1)
        # curr_state = (curr_state - env.observation_space.low) / \
        #                  (env.observation_space.high - env.observation_space.low)

        mu, sigma, action = sess.run([mu_out, sigma_out, act_out], feed_dict={states_: curr_state})

        mu_accel, sigma_accel, action_accel = \
            sess.run([mu_out_accel, sigma_out_accel, act_out_accel], feed_dict={states_: curr_state})

        #DEBUG
        # if throttle:
        #     print( "\tStep: %d - Steering: %.2f - Accel: %.2f" % ( step, action[0][0], action[0][0]))
        # else:
        #     print( "\tStep: %d - Steering: %.2f" % ( step, action[0][0]))

        next_frame, reward, done, _ = env.step([action[0][0], action_accel[0][0]])

        reward_t = reward
        # if reward < 99:
        #     #reward_t = (curr_time - time()) / 10.
        #     reward_t = -1.
        # else:
        #     reward_t = 100.
        ep_reward += reward
        curr_frame = next_frame

        #DEBUG
        step += 1

        states_list.append(curr_state)
        actions_list.append(action[0])
        rewards_list.append(reward_t)
        mu_list.append(mu.reshape(-1,))
        sigma_list.append(sigma.reshape(-1,))

        # Accel implement
        actions_list_accel.append(action_accel[0])
        mu_list_accel.append(mu_accel.reshape(-1,))
        sigma_list_accel.append(sigma_accel.reshape(-1,))

        if done:
            states = np.vstack(states_list)
            actions = np.vstack(actions_list)
            rewards = np.hstack(rewards_list)
            mus = np.hstack(mu_list)
            sigmas = np.hstack(sigma_list)
            # Accel impl
            actions_accel = np.vstack(actions_list_accel)
            mus_accel = np.hstack(mu_list_accel)
            sigmas_accel = np.hstack(sigma_list_accel)

            # Stats for plotting
            ep_scores.append( ep_reward)

            returns = np.zeros_like(rewards)
            rolling = 0

            for i in reversed(range(len(rewards))):
                rolling = rolling * gamma_ + rewards[i]
                returns[i] = rolling

            returns -= np.mean(returns)
            returns /= np.std(returns)

            feed_dict = {states_: states, actions_: actions, returns_: returns.reshape(-1, 1)}

            sess.run(train_value, feed_dict=feed_dict)
            sess.run(train_policy, feed_dict=feed_dict)

            feed_dict_accel = {states_: states,
                actions_accel_: actions_accel,
                returns_: returns.reshape(-1, 1)}

            sess.run(train_value_accel, feed_dict=feed_dict_accel)
            sess.run(train_policy_accel, feed_dict=feed_dict_accel)

            print("\nEpisode : %s," % (i_ep) + \
                "\nMean mu : %.5f, Min mu : %.5f, Max mu : %.5f, Median mu : %.5f" % \
                (np.nanmean(mus), np.min(mus), np.max(mus), np.median(mus)) + \
                "\nMean sigma : %.5f, Min sigma : %.5f, Max sigma : %.5f, Median sigma : %.5f" % \
                (np.nanmean(sigmas), np.min(sigmas), np.max(sigmas), np.median(sigmas)))

            #Accel impl
            print("\nAccel related stats : ," + \
                "\nMean mu : %.5f, Min mu : %.5f, Max mu : %.5f, Median mu : %.5f" % \
                (np.nanmean(mus_accel), np.min(mus_accel), np.max(mus_accel), np.median(mus_accel)) + \
                "\nMean sigma : %.5f, Min sigma : %.5f, Max sigma : %.5f, Median sigma : %.5f" % \
                (np.nanmean(sigmas_accel), np.min(sigmas_accel), np.max(sigmas_accel), np.median(sigmas_accel)) + \
                "\nScores : %.4f, Max reward : %.4f, Min reward : %.4f" % (ep_reward, np.max(rewards), np.min(rewards)))

            #Saving trained model
            if( i_ep % save_every_how_many_ep == 0  and i_ep > 0) or i_ep+1 == num_episodes:
                saver = tf.train.Saver()
                model_file_name = "torcs_a2c_accel_{}@ep_{}_scored_{:.0f}.tfckpt".format( datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3], i_ep, ep_reward)
                #Corresponding pickle file
                stats_file_name = "torcs_a2c_accel_{}@ep_{}_scored_{:.0f}.pickle".format( datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3], i_ep, ep_reward)
                # print( model_file_name)
                save_path = saver.save( sess, save_base_path + model_file_name)
                print("Model saved in path: %s" % save_path)

                #Pickle stats like score and ep
                if saving_stats:
                    # stats_file_name = "torcs_a2c_cont_accel_{}@ep_{}_scored_{:.0f}.pickle".format( datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], i_ep, ep_reward)
                    stats_save_path = stats_base_path + stats_file_name
                    with open( stats_save_path, "wb") as stats_save_file:
                        pickle.dump( ep_scores, stats_save_file)

sess.close()
env.end()
