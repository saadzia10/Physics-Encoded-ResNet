import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import xavier_initializer

from utils import tf_util


class A2C_Agent( object):
    def __init__( self, input_dim=22, hidden_1=100, hidden_2=100,
        lr_actor=1e-3, lr_critic=1e-2, gamma=.95,
        action_low=-1.0, action_high=1.0):

        # Init TF variables
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        # TF Session
        self.sess = tf_util.make_session()

        ## NN setup
        states_ = tf.placeholder(tf.float32, [None, input_dim], name="states_")
        actions_ = tf.placeholder(tf.float32, [None, 1], name="actions_")
        returns_ = tf.placeholder(tf.float32, [None, 1], name="returns_")

        actor_scope = "Actor_Accel"
        with tf.variable_scope(actor_scope):
            self.h_1_act = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
                                name="h_1_act", use_bias=False)
            self.h_2_act = layers.dense(inputs=self.h_1_act, units=hidden_2, kernel_initializer=xavier_initializer(), \
                                name="h_2_act", use_bias=False)
            self.mu_out = layers.dense(inputs=self.h_2_act, units=1, kernel_initializer=xavier_initializer(), \
                                  activation=tf.nn.tanh, name="mu_out", use_bias=False)
            self.sigma_out = layers.dense(inputs=self.h_2_act, units=1, kernel_initializer=xavier_initializer(), \
                                     activation=tf.nn.softplus, name="sigma_out", use_bias=False)
            self.normal_dist = tf.distributions.Normal(loc=self.mu_out, scale=self.sigma_out)
            self.act_out = tf.reshape(self.normal_dist.sample(1), shape=[-1,1])
            self.act_out = tf.clip_by_value(self.act_out, action_low, action_high)

        critic_scope = "Critic_Accel"
        with tf.variable_scope(critic_scope):
            self.h_1_cri = layers.dense(inputs=states_, units=hidden_1, kernel_initializer=xavier_initializer(), \
                                name="h_1_cri", use_bias=False)
            self.h_2_cri = layers.dense(inputs=self.h_1_cri, units=hidden_2, kernel_initializer=xavier_initializer(), \
                                name="h_2_cri", use_bias=False)
            self.v_out = layers.dense(inputs=self.h_2_cri, units=1, activation=None, kernel_initializer=xavier_initializer(), \
                                 name="v_out", use_bias=False)

        self.logprobs = self.normal_dist.log_prob(actions_)

        # for more experiences, add entropy term to loss
        self.entropy = self.normal_dist.entropy()
        self.advantages = returns_ - self.v_out

        # Define Policy Loss and Value loss
        self.policy_loss = tf.reduce_mean(-self.logprobs * self.advantages - 0.01*self.entropy)
        self.value_loss = tf.reduce_mean(tf.square(returns_ - self.v_out))

        self.optimizer_policy = tf.train.AdamOptimizer(learning_rate=lr_actor)
        self.optimizer_value = tf.train.AdamOptimizer(learning_rate=lr_critic)

        self.train_policy = self.optimizer_policy.minimize(self.policy_loss, \
                                                 var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Actor_Accel"))
        self.train_value = self.optimizer_value.minimize(self.value_loss, \
                                               var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "Critic_Accel"))

    def predict( self, curr_state):
        mu, sigma, action = self.sess.run([self.mu_out, self.sigma_out, self.act_out], feed_dict={states_: curr_state})

        return mu, sigma, action

    def optimize( self ,states, actions, returns):
        feed_dict = {states_: states, actions_: actions, returns_: returns.reshape(-1, 1)}

        self.sess.run(train_value, feed_dict=feed_dict)
        self.sess.run(train_policy, feed_dict=feed_dict)

    def load(self):
        pass

    def save():
        pass
