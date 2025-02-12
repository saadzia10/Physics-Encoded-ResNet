import argparse
import sys
import optax
from functools import partial
import torch
import numpy as np
import time
import jax
import jax.numpy as jnp
import matplotlib as mp
import haiku as hk
import dill as pickle

mp.use("Qt5Agg")
mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import deep_lagrangian_networks.jax_DeLaN_humanoid_model as delan
from deep_lagrangian_networks.replay_memory import ReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env, activations

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'


train_file_path = "./humanoid/humanoid.csv"  # File created in earlier processing step
test_file_path = "./humanoid/humanoid.csv" #"./humanoid/humanoid_test.csv"

# Compute derivatives for velocities and accelerations
def compute_derivatives(data, dt):
    velocities = np.gradient(data, dt, axis=0)
    accelerations = np.gradient(velocities, dt, axis=0)
    return velocities, accelerations


# Load generalized coordinates from preprocessed dataset
def load_generalized_coordinates(data, dt):
    q = np.array(data)

    # Compute velocities (q_dot) and accelerations (q_ddot)
    q_dot, q_ddot = compute_derivatives(data, dt)

    return q, q_dot, q_ddot


def load_humanoid_dataset(file_path, dt=0.01):
    # Implement data loading from your humanoid dataset
    data = pd.read_csv(file_path)
    columns = [x for x in data.columns if "joint" in x and "Finger" in x]
    q, q_dot, q_ddot = load_generalized_coordinates(data[columns].values, dt)
    return q, q_dot, q_ddot


if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[False, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[4, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    parser.add_argument("-d", nargs=1, type=str, required=False, default=['char', ], help="Dataset")
    parser.add_argument("-t", nargs=1, type=str, required=False, default=['structured', ], help="Lagrangian Type")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())
    rng_key = jax.random.PRNGKey(seed)

    dataset = str(parser.parse_args().d[0])
    model_id = str(parser.parse_args().t[0])

    # Construct Hyperparameters:
    if model_id == "structured":
        lagrangian_type = delan.structured_lagrangian_fn

    elif model_id == "black_box":
        lagrangian_type = delan.blackbox_lagrangian_fn

    else:
        raise ValueError

    hyper = {
        'dataset': dataset,
        'n_width': 64,
        'n_depth': 2,
        'n_minibatch': 512,
        'diagonal_epsilon': 0.1,
        'diagonal_shift': 2.0,
        'activation': 'tanh',
        'learning_rate': 1.e-04,
        'weight_decay': 1.e-5,
        'max_epoch': int(2.5 * 1e3) if dataset == "uniform" else int(10 * 1e3),
        'lagrangian_type': lagrangian_type,
        }

    model_id = "black_box"
    if hyper['lagrangian_type'].__name__ == 'structured_lagrangian_fn':
        model_id = "structured"

    if load_model:
        with open(f"data/delan_models/delan_{model_id}_{hyper['dataset']}_seed_{seed}.jax", 'rb') as f:
            data = pickle.load(f)

        hyper = data["hyper"]
        params = data["params"]

    else:
        params = None

    train_qp, train_qv, train_qa = load_humanoid_dataset(train_file_path)
    n_dof = train_qp.shape[1]

    test_qp, test_qv, test_qa = load_humanoid_dataset(test_file_path)

    # Set memory dimensions
    mem_dim = ((n_dof,), (n_dof,), (n_dof,))  # Adjust to match your data
    mem = ReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_qp, train_qv, train_qa])


    print("\n\n################################################")
    print("Humanoid Motion:")
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Deep Lagrangian Networks (DeLaN):\n")

    # Construct DeLaN:
    t0 = time.perf_counter()

    lagrangian_fn = hk.transform(partial(
        hyper['lagrangian_type'],
        n_dof=n_dof,
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
    ))

    q, qd, qdd = [jnp.array(x) for x in next(iter(mem))]
    rng_key, init_key = jax.random.split(rng_key)

    # Initialize Parameters:
    if params is None:
        params = lagrangian_fn.init(init_key, q[0], qd[0])

    # Trace Model:
    lagrangian = lagrangian_fn.apply
    delan_model = jax.jit(partial(delan.dynamics_model, lagrangian=lagrangian, n_dof=n_dof))

    _ = delan_model(params, None, q[:1], qd[:1], qdd[:1])
    t_build = time.perf_counter() - t0
    print(f"DeLaN Build Time     = {t_build:.2f}s")

    # Generate & Initialize the Optimizer:
    t0 = time.perf_counter()

    optimizer = optax.adamw(
        learning_rate=hyper['learning_rate'],
        weight_decay=hyper['weight_decay']
    )

    opt_state = optimizer.init(params)
    loss_fn = partial(
        delan.loss_fn,
        lagrangian=lagrangian,
        n_dof=n_dof,
        norm_qdd=jnp.var(train_qa, axis=0)
    )

    def update_fn(params, opt_state, q, qd, qdd):

        (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, qd, qdd)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, logs

    update_fn = jax.jit(update_fn)
    _, _, logs = update_fn(params, opt_state, q[:1], qd[:1], qdd[:1])

    t_build = time.perf_counter() - t0
    print(f"Optimizer Build Time = {t_build:.2f}s")

    # Start Training Loop:
    t0_start = time.perf_counter()

    print("")
    epoch_i = 0
    while epoch_i < hyper['max_epoch'] and not load_model:
        n_batches = 0
        logs = jax.tree_map(lambda x: x * 0.0, logs)

        for data_batch in mem:
            t0_batch = time.perf_counter()

            q, qd, qdd = [jnp.array(x) for x in data_batch]
            params, opt_state, batch_logs = update_fn(params, opt_state, q, qd, qdd)

            # Update logs:
            n_batches += 1
            logs = jax.tree_map(lambda x, y: x + y, logs, batch_logs)
            print(f"Loss = {logs['loss']:.1e}", end=", ")
            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        epoch_i += 1
        logs = jax.tree_map(lambda x: x/n_batches, logs)

        if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
            print(f"Loss = {logs['loss']:.1e}", end=", ")
            print(f"For = {logs['forward_mean']:.1e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.1e}", end=", ")
            print(f"Power = {logs['energy_mean']:.1e} \u00B1 {1.96 * np.sqrt(logs['energy_var']):.1e}")

    # Save the Model:
    if save_model:
        with open(f"data/delan_models/delan_{model_id}_{hyper['dataset']}_seed_{seed}.jax", "wb") as file:
            pickle.dump(
                {"epoch": epoch_i,
                 "hyper": hyper,
                 "params": params,
                 "seed": seed},
                file)

    print("\n################################################")
    print("Evaluating DeLaN:")

