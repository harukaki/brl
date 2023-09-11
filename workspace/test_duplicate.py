import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any, Literal, Type
import distrax
import pgx
import pgx.bridge_bidding as bb
from utils import (
    auto_reset,
    single_play_step_vs_policy_in_bridge_bidding,
    single_play_step_vs_DeepMind_sl_model,
    single_play_step_vs_DeepMind_sl_model_in_deterministic,
    duplicate_play_step_vs_DeepMind_sl_model,
)
import time
import os
import random
from pprint import pprint


import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb

from duplicate import duplicate_step
from models import ActorCritic, DeepMind_sl_net_fn


print(jax.default_backend())
print(jax.local_devices())


def rl_forward_fn(x):
    net = ActorCritic(38, activation="relu", model="DeepMind")
    logits, value = net(x)
    return logits, value


rl_forward_pass = hk.without_apply_rng(hk.transform(rl_forward_fn))
# sl_forward_pass = hk.without_apply_rng(hk.transform(DeepMind_sl_net_fn))
DeepMind_sl_model_param = pickle.load(
    open("bridge_bidding_sl_networks/params-290000.pkl", "rb")
)
sl_params = pickle.load(open("sl_params/params-290000.pkl", "rb"))
sl_forward_pass = hk.without_apply_rng(hk.transform(rl_forward_fn))


def wrapped_sl_forward_pass_apply(observation):
    return sl_forward_pass.apply(DeepMind_sl_model_param, observation)


"""
def wraped_sl_forward_pass_apply(observation):
    return sl_forward_pass.apply(DeepMind_sl_model_param, observation)
"""

eval_env = bb.BridgeBidding("dds_results/test_000.npy")


def _get(x, i):
    return x[i]


def duplicate_evaluate(params, rng_key):
    num_eval_envs = 10000
    step_fn = duplicate_step(eval_env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_eval_envs)
    state = jax.vmap(eval_env.init)(subkeys)
    # state.save_svg("svg/eval_init.svg")
    cum_return = jnp.zeros(num_eval_envs)
    get_fn = _get
    i = 0
    states = []
    table_a_reward = state.rewards
    has_duplicate_result = jnp.zeros(num_eval_envs, dtype=jnp.bool_)

    def cond_fn(tup):
        (
            state,
            table_a_reward,
            has_duplicate_result,
            _,
            _,
        ) = tup
        return ~state.terminated.all()

    def rl_make_action(state):
        logits_old, value = rl_forward_pass.apply(params, state.observation)  # DONE
        mask_logits = jnp.finfo(np.float64).min * (~state.legal_action_mask)
        logits = logits_old + mask_logits
        pi = distrax.Categorical(logits=logits)
        return (
            pi.mode(),
            pi.probs,
            logits,
            state.legal_action_mask,
            logits_old,
            mask_logits,
        )

    """
    def sl_make_action(state):
        logits_old = wraped_sl_forward_pass_apply(
            state.observation.astype(jnp.float32)
        )
        mask_logits = jnp.finfo(jnp.float32).min * (~state.legal_action_mask)
        logits = logits_old + mask_logits
        pi = jnp.exp(logits)
        # pi = distrax.Categorical(logits=logits)
        # return , pi
        return (
            jnp.argmax(pi),
            pi,
            logits,
            state.legal_action_mask,
            logits_old,
            mask_logits,
        )
    """

    def sl_make_action(state):
        logits_old, value = rl_forward_pass.apply(sl_params, state.observation)  # DONE
        mask_logits = jnp.finfo(np.float64).min * (~state.legal_action_mask)
        logits = logits_old + mask_logits
        pi = distrax.Categorical(logits=logits)
        return (
            pi.mode(),
            pi.probs,
            logits,
            state.legal_action_mask,
            logits_old,
            mask_logits,
        )

    def make_action(state):
        return jax.lax.cond(
            (state.current_player == 0) | (state.current_player == 1),
            lambda: rl_make_action(state),
            lambda: sl_make_action(state),
        )

    def loop_fn(tup):
        state, table_a_reward, has_duplicate_result, cum_return, rng_key = tup
        action, probs, logits, mask, logits_old, mask_logits = jax.vmap(make_action)(
            state
        )
        """
        print(f"current_player {state.current_player}")
        print(f"actor {jax.vmap(check_actor)(state)}")
        print(f"action {action}")
        print(f"pi {probs}")
        print(f"pi {probs.dtype}")
        print(f"logits {logits}")
        print(f"logits {logits.dtype}")
        print(f"logits {logits.shape}")
        print(f"mask {mask}")
        print(f"mask {mask.dtype}")
        print(f"mask {mask.shape}")
        print(f"mask logits {mask_logits}")
        print(f"mask logits {mask_logits.dtype}")
        print(f"mask logits {mask_logits.shape}")
        print(f"logits_old {logits_old}")
        print(f"logits_old {logits_old.dtype}")
        print(f"logits_old {logits_old.shape}")
        """
        rng_key, _rng = jax.random.split(rng_key)
        (state, table_a_reward, has_duplicate_result) = jax.vmap(step_fn)(
            state, action, table_a_reward, has_duplicate_result
        )
        cum_return = cum_return + jax.vmap(get_fn)(
            state.rewards, jnp.zeros_like(state.current_player)
        )
        return state, table_a_reward, has_duplicate_result, cum_return, rng_key

    (
        state,
        table_a_reward,
        has_duplicate_result,
        cum_return,
        _,
    ) = jax.lax.while_loop(
        cond_fn,
        loop_fn,
        (state, table_a_reward, has_duplicate_result, cum_return, rng_key),
    )
    """
    def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
            # print(val[0])

            print(val[1], val[2], val[3], val[4])
        return val

    (
        state,
        table_a_reward,
        has_duplicate_result,
        cum_return,
        _,
    ) = while_loop(
        cond_fn,
        loop_fn,
        (state, table_a_reward, has_duplicate_result, cum_return, rng_key),
    )
    """
    # state.save_svg("svg/duplicate.svg")

    return cum_return.mean(), cum_return


if __name__ == "__main__":
    # INIT NETWORK
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + eval_env.observation_shape)
    params = rl_forward_pass.init(_rng, init_x)
    params = pickle.load(open("sl_params/params-200000.pkl", "rb"))
    cum_return_mean, cum_return = duplicate_evaluate(params, _rng)
    print(cum_return_mean)
    print(cum_return)
