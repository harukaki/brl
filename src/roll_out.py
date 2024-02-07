import jax
import distrax
from typing import NamedTuple, Any, Literal
import jax.numpy as jnp
import numpy as np
from src.utils import (
    single_play_step_two_policy_commpetitive,
    single_play_step_free_run,
)
from pgx.experimental import auto_reset


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def make_roll_out(config, env, actor_forward_pass, opp_forward_pass):
    def make_policy(config):
        if config["ACTOR_ILLEGAL_ACTION_MASK"]:

            def masked_policy(mask, logits):
                logits = logits + jnp.finfo(np.float64).min * (~mask)
                pi = distrax.Categorical(logits=logits)
                return pi

            return masked_policy
        elif config["ACTOR_ILLEGAL_ACTION_PENALTY"]:

            def no_masked_policy(mask, logits):
                pi = distrax.Categorical(logits=logits)
                return pi

            return no_masked_policy

    if config["GAME_MODE"] == "competitive":
        make_step_fn = single_play_step_two_policy_commpetitive
    elif config["GAME_MODE"] == "free-run":
        make_step_fn = single_play_step_free_run
        opp_forward_pass = None
        opp_params = None
    policy = make_policy(config)

    def roll_out(runner_state, opp_params):
        step_fn = make_step_fn(
            step_fn=auto_reset(env.step, env.init),
            actor_forward_pass=actor_forward_pass,
            actor_params=runner_state[0],
            opp_forward_pass=opp_forward_pass,
            opp_params=opp_params,
        )

        def _get(x, i):
            return x[i]

        get_fn = _get

        def _env_step(runner_state, unused):
            (
                params,
                opt_state,
                env_state,
                last_obs,
                terminated_count,
                rng,
            ) = runner_state  # DONE
            actor = env_state.current_player
            logits, value = actor_forward_pass.apply(
                params,
                last_obs.astype(jnp.float32),
            )  # DONE
            rng, _rng = jax.random.split(rng)
            mask = env_state.legal_action_mask
            pi = policy(mask, logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state = step_fn(env_state, action, _rng)
            terminated_count += jnp.sum(env_state.terminated)
            transition = Transition(
                env_state.terminated,
                action,
                value,
                jax.vmap(get_fn)(env_state.rewards / config["REWARD_SCALE"], actor),
                log_prob,
                last_obs,
                mask,
            )
            runner_state = (
                params,
                opt_state,
                env_state,
                env_state.observation,
                terminated_count,
                rng,
            )  # DONE
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )
        return runner_state, traj_batch

    return roll_out
