import os
from pathlib import Path
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

tf.config.set_visible_devices([], 'GPU')

num_states = 24
num_actions = 1
upper_bound = 1
lower_bound = 200

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model, target_actor,
            target_critic, gamma, critic_optimizer, actor_optimizer
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # Clip-by-value on all trainable gradients
        # critic_grad = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0))
        #              for grad in critic_grad]


        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # Clip-by-value on all trainable gradients
        # actor_grad = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0))
        #                for grad in actor_grad]

        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        return critic_grad, actor_grad

    # We compute the loss and update parameters
    def learn(self, actor_model, critic_model,
                    target_actor, target_critic, gamma, critic_optimizer, actor_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        critic_grad, actor_grad, critic_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model,
                    target_actor, target_critic, gamma, critic_optimizer, actor_optimizer)

        # print("Actor Grad: ", [np.median(g.numpy()) for g in actor_grad])
        # print("Critic Grad: ", [np.median(g.numpy()) for g in critic_grad])
        # return critic_loss, critic_grad



class Trainer:
    def __init__(self):
        self.save_path = Path("models_ckpt")
        if not self.save_path.exists():
            os.makedirs(self.save_path.as_posix())

        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.total_episodes = 100
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        self.buffer = Buffer(50000, 64)

    def save_weights(self):
        print("Saving weights")
        self.actor_model.save_weights(self.save_path.joinpath("actor_model.h5").as_posix())
        self.critic_model.save_weights(self.save_path.joinpath("critic_model.h5").as_posix())

    def load_weights(self):
        print("Loading weights")
        self.actor_model.load_weights(self.save_path.joinpath("actor_model.h5").as_posix())
        self.critic_model.load_weights(self.save_path.joinpath("critic_model.h5").as_posix())

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @staticmethod
    @tf.function
    def update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    @staticmethod
    def get_actor():
        # Initialize weights between -3e-3 and 3-e3
        # last_init = tf.random_uniform_initializer(minval=-5., maxval=5.)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(8, activation="relu")(inputs)
        out = layers.Dense(4, activation="relu")(out)
        outputs = layers.Dense(1, activation="relu")(out)
        # outputs = layers.Dense(1, activation="relu")(out)

        model = tf.keras.Model(inputs, outputs)
        return model

    def analyze_actor(self):
        return [np.median(l) for l in self.actor_model.get_weights()]

    def analyze_critic(self):
        return [np.median(l) for l in self.critic_model.get_weights()]

    @staticmethod
    def get_critic():
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(state_input)
        state_out = layers.Dense(16, activation="linear", kernel_initializer="he_normal")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(8, activation="linear", kernel_initializer="he_normal")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(8, activation="relu", kernel_initializer="he_normal")(concat)
        out = layers.Dense(4, activation="relu", kernel_initializer="he_normal")(concat)
        # out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="linear", kernel_initializer="he_normal")(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state)) * 100.
        print("Actor weights: ", self.analyze_actor())
        print("Critic weights: ", self.analyze_critic())
        print("L: ", np.squeeze(sampled_actions))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return [np.squeeze(legal_action)]

