import numpy as np
import copy
from gym_torcs_wrpd import TorcsEnv
from utils.rewarder import Rewarder
from scripts import snakeoil3_gym_raceconfig as snakeoil3


class PPEnv(TorcsEnv):
    def __init__(self, vision=False, throttle=True, gear_change=False, brake_change=True, race_config_path=None, render=False):
        super().__init__(vision=vision, render=render, throttle=throttle, gear_change=gear_change, race_config_path=race_config_path)
        self.brake_change = brake_change
        self.rewarder = Rewarder()

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.brake_change is True:
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            print(u[3])
            torcs_action.update({'gear': u[3]})

        return torcs_action

    def step(self, u, continuous=True, normalize=True):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(self.disc_to_continuous(u)) if not continuous else self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            # action_torcs['gear'] = 1

            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6


        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs, normalize=normalize)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])

        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])
        reward = self.rewarder.calculate_reward(obs, obs_pre, action_torcs)

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return obs, self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False, normalize=True):
        #print("Reset")
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3101, vision=self.vision,
            process_id=self.torcs_process_id,
            race_config_path=self.race_config_path)  #Open new UDP in vtorcs

        self.client.MAX_STEPS = np.inf

        client = self.client

        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs, normalize=normalize)

        self.last_u = None

        self.initial_reset = False

        # THe newly created TOrcs PID is also reattached to the Gym Torcs Env
        # This should be temporary ... but only time knows
        self.torcs_process_id = self.client.torcs_process_id

        return obs, self.get_obs()

