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

env = gym.make("MountainCarContinuous-v0")

# hyperparameters
hidden_1 = 80
hidden_2 = 80
lr_actor = 1e-3
lr_critic = 1e-2
gamma_ = 0.95
frame = 0
num_episodes = 100
episode = 0
input_dim = env.observation_space.shape[0]

tf.reset_default_graph()

# tensorflow graph
states_ = tf.placeholder(tf.float32, [None, input_dim])
actions_ = tf.placeholder(tf.float32, [None, 1])
returns_ = tf.placeholder(tf.float32, [None, 1])

actor_scope = "Actor"
with tf.variable_scope(actor_scope):
    h_1_act = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
                       activation=tf.nn.selu, name="h_1_act", use_bias=False)
    h_2_act = layers.dense(inputs=h_1_act, units=hidden_2, kernel_initializer=xavier_initializer(), \
                       activation=tf.nn.selu, name="h_2_act", use_bias=False)
    mu_out = layers.dense(inputs=h_2_act, units=1, kernel_initializer=xavier_initializer(), \
                          activation=tf.nn.tanh, name="mu_out", use_bias=False)
    sigma_out = layers.dense(inputs=h_2_act, units=1, kernel_initializer=xavier_initializer(), \
                             activation=tf.nn.softplus, name="sigma_out", use_bias=False)
    normal_dist = tf.distributions.Normal(loc=mu_out, scale=sigma_out)
    act_out = tf.reshape(normal_dist.sample(1), shape=[-1,1])
    act_out = tf.clip_by_value(act_out, env.action_space.low, env.action_space.high)

critic_scope = "Critic"
with tf.variable_scope(critic_scope):
    h_1_cri = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
                       activation=tf.nn.selu, name="h_1_cri", use_bias=False)
    h_2_cri = layers.dense(inputs=h_1_cri, units=hidden_2, kernel_initializer=xavier_initializer(), \
                       activation=tf.nn.selu, name="h_2_cri", use_bias=False)
    v_out = layers.dense(inputs=h_2_cri, units=1, activation=None, kernel_initializer=xavier_initializer(), \
                         name="v_out", use_bias=False)

logprobs = normal_dist.log_prob(actions_)

# for more experiences, add entropy term to loss
entropy = normal_dist.entropy()
advantages = returns_ - v_out

# Define Policy Loss and Value loss
policy_loss = tf.reduce_mean(-logprobs * advantages - 0.01*entropy)
value_loss = tf.reduce_mean(tf.square(returns_ - v_out))

optimizer_policy = tf.train.AdamOptimizer(learning_rate=lr_actor)
optimizer_value = tf.train.AdamOptimizer(learning_rate=lr_critic)

train_policy = optimizer_policy.minimize(policy_loss, \
                                         var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Actor"))
train_value = optimizer_value.minimize(value_loss, \
                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Critic"))

# Configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# session
sess = tf.Session(config=config)

# initializing all variables
init = tf.global_variables_initializer()
sess.run(init)

# A2C algorithm
for i_ep in range(num_episodes):
    states_list = []
    actions_list = []
    rewards_list = []
    mu_list = []
    sigma_list = []
    ep_reward = 0
    ep_rewards = []
    curr_frame = env.reset()
    done = False
    curr_time = time()
    while done is False:
        curr_state = curr_frame.reshape(1, -1)
        curr_state = (curr_state - env.observation_space.low) / \
                         (env.observation_space.high - env.observation_space.low)
        mu, sigma, action = sess.run([mu_out, sigma_out, act_out], feed_dict={states_: curr_state})
        next_frame, reward, done, _ = env.step(action)
        if reward < 99:
            #reward_t = (curr_time - time()) / 10.
            reward_t = -1.
        else:
            reward_t = 100.
        ep_reward += reward
        curr_frame = next_frame

        states_list.append(curr_state)
        actions_list.append(action)
        rewards_list.append(reward_t)
        mu_list.append(mu.reshape(-1,))
        sigma_list.append(sigma.reshape(-1,))

        if done:
            states = np.vstack(states_list)
            actions = np.vstack(actions_list)
            rewards = np.hstack(rewards_list)
            mus = np.hstack(mu_list)
            sigmas = np.hstack(sigma_list)

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

            print("\nEpisode : %s," % (i_ep) + \
                   "\nMean mu : %.5f, Min mu : %.5f, Max mu : %.5f, Median mu : %.5f" % \
                       (np.nanmean(mus), np.min(mus), np.max(mus), np.median(mus)) + \
                   "\nMean sigma : %.5f, Min sigma : %.5f, Max sigma : %.5f, Median sigma : %.5f" % \
                       (np.nanmean(sigmas), np.min(sigmas), np.max(sigmas), np.median(sigmas)) + \
                   "\nScores : %.4f, Max reward : %.4f, Min reward : %.4f" % (ep_reward, np.max(rewards), np.min(rewards)))
