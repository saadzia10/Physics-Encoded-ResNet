"""
Helpers for scripts like run_atari.py.
Repurposed for TORCS
"""

import os
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
# from baselines.common.atari_wrappers import wrap_deepmind
from scripts.torqs_wrappers import wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from gym_torcs_wrpd import TorcsEnv

### # REVIEW: Default Race config cannot be none
def make_torcs_env(num_env=1, seed=42, wrapper_kwargs=None, start_index=0,
    vision=False, throttle=False, gear_change=False, race_config_path=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Torqs.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank, vision=False, throttle=False, gear_change=False,
        race_config_path=None): # pylint: disable=C0111
        def _thunk():
            #### REVIEW: Find out the impact of the rank of an env
            env = TorcsEnv(vision=vision, throttle=throttle, gear_change=throttle,
            race_config_path=race_config_path)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)

    return SubprocVecEnv([make_env( rank=( i+start_index),
        vision=vision, throttle=throttle, gear_change=throttle,
        race_config_path=race_config_path)
        for i in range(num_env)])

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def torcs_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    #Duh
    # parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    #Probably useless
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser
