import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

# from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Literal, Type
import distrax
import pgx
import pgx.bridge_bidding as bb
from utils import (
    auto_reset,
    single_play_step_vs_policy_in_backgammon,
    single_play_step_vs_policy_in_two,
    normal_step,
    single_play_step_vs_policy_in_sparrow_mahjong,
    single_play_step_vs_policy_in_bridge_bidding,
    single_play_step_competitive,
)
import time
import os

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb

env = bb.BridgeBidding(
    dds_hash_table_path="dds_hash_table/dds_hash_table_0000k.npy"
)


class DeepMind_sl_model(hk.Module):
    """Haiku module for our network."""

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def __call__(self, x):
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.action_dim)(x)
        x = jax.nn.log_softmax(x)
        return x


def sl_forward_pass(x):
    net = DeepMind_sl_model(env.num_actions)
    logits = net(x)
    return logits


sl_forward_pass = hk.without_apply_rng(hk.transform(sl_forward_pass))

params = pickle.load(
    open("bridge_bidding_sl_networks/params-240000.pkl", "rb")
)
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
init_x = jnp.zeros((1,) + env.observation_shape)
print(init_x)
model = sl_forward_pass.init(subkey, init_x)
print(model)
# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))


N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)
state: pgx.State = init(keys)
state = state.replace(
    _vul_NS=jnp.zeros(N, dtype=jnp.bool_),
    _vul_EW=jnp.zeros(N, dtype=jnp.bool_),
)  # wbridge5のデータセットはノンバルのみ
print(state)
teamB_forward_pass = sl_forward_pass
teamB_model_params = pickle.load(
    open("bridge_bidding_sl_networks/params-290000.pkl", "rb")
)
print(teamB_model_params)
i = 0
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    logits = teamB_forward_pass.apply(
        teamB_model_params, state.observation.astype(jnp.float32)
    )
    pi = distrax.Categorical(logits=logits)
    action = pi.sample(seed=key)
    print(action)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    state.save_svg(f"test/{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.rewards}")
    i += 1
state.save_svg(f"{i:04d}.svg")
