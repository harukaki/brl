import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any, Literal, Type
import distrax
import os
from pgx.bridge_bidding import BridgeBidding
import pickle
import wandb
from src.duplicate import duplicate_step, Table_info
from src.models import make_forward_pass
from src.utils import single_play_step_two_policy_commpetitive_deterministic
from pprint import pprint


def make_simple_evaluate(config):
    eval_env = BridgeBidding("dds_results/test_000.npy")
    # eval_env = BridgeBidding("workspace/100_hash.npy")
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    opp_forward_pass = make_forward_pass(
        activation=config["EVAL_OPP_ACTIVATION"],
        model_type=config["EVAL_OPP_MODEL_TYPE"],
    )
    opp_params = pickle.load(open(config["EVAL_OPP_MODEL_PATH"], "rb"))
    num_eval_envs = config["NUM_EVAL_ENVS"]

    def get_fn(x, i):
        return x[i]

    def simple_evaluate(actor_params, rng):
        step_fn = single_play_step_two_policy_commpetitive_deterministic(
            step_fn=eval_env.step,
            actor_params=actor_params,
            actor_forward_pass=actor_forward_pass,
            opp_params=opp_params,
            opp_forward_pass=opp_forward_pass,
        )
        rng_key, sub_key = jax.random.split(rng)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)
        R = jnp.zeros(num_eval_envs)

        def cond_fn(tup):
            state, _, _ = tup
            return ~state.terminated.all()

        def loop_fn(tup):
            state, R, rng_key = tup
            actor = state.current_player
            logits, value = actor_forward_pass.apply(actor_params, state.observation)
            logits = logits + jnp.finfo(np.float64).min * (~state.legal_action_mask)
            pi = distrax.Categorical(logits=logits)
            action = pi.mode()
            rng_key, _rng = jax.random.split(rng_key)
            state = step_fn(state, action, _rng)
            R = R + jax.vmap(get_fn)(state.rewards, actor)
            return state, R, rng_key

        state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
        return R.mean()

    return simple_evaluate


def make_simple_duplicate_evaluate(config):
    eval_env = BridgeBidding("dds_results/test_000.npy")
    team1_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    team2_forward_pass = team1_forward_pass
    num_eval_envs = config["NUM_PRIORITIZED_ENVS"]

    def duplicate_evaluate(
        team1_params,
        team2_params,
        # num_eval_envs,
        rng_key,
    ):
        step_fn = duplicate_step(eval_env.step)
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)
        # state.save_svg("svg/eval_init.svg")
        cum_return = jnp.zeros(num_eval_envs)
        table_a_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )
        table_b_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )

        cum_return = jnp.zeros(num_eval_envs)
        count = 0

        def get_fn(x, i):
            return x[i]

        def cond_fn(tup):
            (state, _, _, _, _, _) = tup
            return ~state.terminated.all()

        def actor_make_action(state):
            logits, value = team1_forward_pass.apply(
                team1_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float64).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), pi.probs)

        def opp_make_action(state):
            logits, value = team2_forward_pass.apply(
                team2_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float64).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), pi.probs)

        def make_action(state):
            return jax.lax.cond(
                (state.current_player == 0) | (state.current_player == 1),
                lambda: actor_make_action(state),
                lambda: opp_make_action(state),
            )

        def loop_fn(tup):
            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,
            ) = tup
            (action, pi_probs) = jax.vmap(make_action)(state)
            rng_key, _rng = jax.random.split(rng_key)
            (state, table_a_info, table_b_info) = jax.vmap(step_fn)(
                state, action, table_a_info, table_b_info
            )
            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards, jnp.zeros_like(state.current_player)
            )
            count += 1
            # state.save_svg(f"svg/{count}.svg")
            return (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,
            )

        (
            state,
            table_a_info,
            table_b_info,
            cum_return,
            _,
            count,
        ) = jax.lax.while_loop(
            cond_fn,
            loop_fn,
            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,
            ),
        )
        # state.save_svg("svg/duplicate.svg")
        std_error = jnp.std(cum_return, ddof=1) / jnp.sqrt(len(cum_return))
        win_rate = jnp.sum(cum_return > 0) / num_eval_envs
        log_info = (cum_return.mean(), std_error, win_rate)
        return log_info, table_a_info, table_b_info

    return duplicate_evaluate


def make_evaluate(config, duplicate=False):
    eval_env = BridgeBidding("dds_results/test_000.npy")
    # eval_env = BridgeBidding("workspace/100_hash.npy")
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    opp_forward_pass = make_forward_pass(
        activation=config["EVAL_OPP_ACTIVATION"],
        model_type=config["EVAL_OPP_MODEL_TYPE"],
    )
    opp_params = pickle.load(open(config["EVAL_OPP_MODEL_PATH"], "rb"))
    num_eval_envs = config["NUM_EVAL_ENVS"]

    if config["GAME_MODE"] == "competitive":
        opp_forward_pass = make_forward_pass(
            activation=config["EVAL_OPP_ACTIVATION"],
            model_type=config["EVAL_OPP_MODEL_TYPE"],
        )
        opp_params = pickle.load(open(config["EVAL_OPP_MODEL_PATH"], "rb"))

        def opp_make_action(state):
            logits, value = opp_forward_pass.apply(
                opp_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float64).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), pi.probs)

    elif config["GAME_MODE"] == "free-run":

        def opp_make_action(state):
            pi_probs = jnp.zeros(38).at[0].set(True)
            return (jnp.int32(0), pi_probs)

    def get_fn(x, i):
        return x[i]

    def evaluate(actor_params, rng_key):
        step_fn = eval_env.step
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)

        cum_return = jnp.zeros(num_eval_envs)
        actor_total_illegal_action_probs = jnp.zeros(num_eval_envs)
        opp_total_illegal_action_probs = jnp.zeros(num_eval_envs)
        actor_step_count = jnp.zeros(num_eval_envs)
        opp_step_count = jnp.zeros(num_eval_envs)
        actor_bid = jnp.zeros((num_eval_envs, 35))
        opp_bid = jnp.zeros((num_eval_envs, 35))
        actor_pass_count = jnp.zeros(num_eval_envs)
        opp_pass_count = jnp.zeros(num_eval_envs)

        def cond_fn(tup):
            (state, _, _, _) = tup
            return ~state.terminated.all()

        def rl_make_action(state):
            logits_old, value = actor_forward_pass.apply(
                actor_params, state.observation
            )  # DONE
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

        def pass_act(state):
            pi_probs = jnp.zeros(38).at[0].set(0)
            return (
                jnp.int32(0),
                pi_probs,
                jnp.zeros(38, dtype=jnp.float32),
                state.legal_action_mask,
                jnp.zeros(38, dtype=jnp.float32),
                jnp.zeros(38, dtype=jnp.float32),
            )

        def sl_make_action(state):
            logits_old, value = opp_forward_pass.apply(
                opp_params, state.observation
            )  # DONE
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

        def update_log_info(logits_old, mask, current_player, action, log_info):
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ) = log_info
            illegal_pi = distrax.Categorical(logits=logits_old)
            illegal_action_prob = jnp.dot(illegal_pi.probs, ~mask)
            actor_total_illegal_action_probs = jax.lax.cond(
                current_player < 2,
                lambda: actor_total_illegal_action_probs + illegal_action_prob,
                lambda: actor_total_illegal_action_probs,
            )
            opp_total_illegal_action_probs = jax.lax.cond(
                current_player >= 2,
                lambda: opp_total_illegal_action_probs + illegal_action_prob,
                lambda: opp_total_illegal_action_probs,
            )
            actor_step_count = jax.lax.cond(
                current_player < 2,
                lambda: actor_step_count + 1,
                lambda: actor_step_count,
            )
            opp_step_count = jax.lax.cond(
                current_player >= 2,
                lambda: opp_step_count + 1,
                lambda: opp_step_count,
            )
            actor_bid = jax.lax.cond(
                (current_player < 2) & (action >= 3),
                lambda: actor_bid.at[action - 3].set(1),
                lambda: actor_bid,
            )
            opp_bid = jax.lax.cond(
                (current_player >= 2) & (action >= 3),
                lambda: opp_bid.at[action - 3].set(1),
                lambda: opp_bid,
            )
            actor_pass_count = jax.lax.cond(
                (current_player < 2) & (action == 0),
                lambda: actor_pass_count + 1,
                lambda: actor_pass_count,
            )
            opp_pass_count = jax.lax.cond(
                (current_player >= 2) & (action == 0),
                lambda: opp_pass_count + 1,
                lambda: opp_pass_count,
            )
            return (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            )

        def make_step_log(state, logits_old, mask, current_player, action, log_info):
            return jax.lax.cond(
                state.terminated,
                lambda: log_info,
                lambda: update_log_info(
                    logits_old, mask, current_player, action, log_info
                ),
            )

        def loop_fn(tup):
            (state, cum_return, log_info, rewards) = tup

            # current_player_position = _player_position(state.current_player, state)

            action, probs, logits, mask, logits_old, mask_logits = jax.vmap(
                make_action
            )(state)
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ) = jax.vmap(make_step_log)(
                state, logits_old, mask, state.current_player, action, log_info
            )

            state = jax.vmap(step_fn)(state, action)
            rewards += state.rewards

            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards, jnp.zeros_like(state.current_player)
            )
            # print(f"cum_return: {cum_return}")
            return (
                state,
                cum_return,
                (
                    actor_total_illegal_action_probs,
                    opp_total_illegal_action_probs,
                    actor_step_count,
                    opp_step_count,
                    actor_bid,
                    opp_bid,
                    actor_pass_count,
                    opp_pass_count,
                ),
                rewards,
            )

        rewards = jnp.zeros_like(state.rewards)
        (
            state,
            cum_return,
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ),
            rewards,
        ) = jax.lax.while_loop(
            cond_fn,
            loop_fn,
            (
                state,
                cum_return,
                (
                    actor_total_illegal_action_probs,
                    opp_total_illegal_action_probs,
                    actor_step_count,
                    opp_step_count,
                    actor_bid,
                    opp_bid,
                    actor_pass_count,
                    opp_pass_count,
                ),
                rewards,
            ),
        )

        def make_terminated_log(state, cum_return):
            return jax.lax.cond(
                (state._last_bidder == -1)
                & (state._last_bid == -1)
                & (state._pass_num == 4),
                lambda: (
                    jnp.bool_(True),
                    (
                        jnp.zeros(35),
                        jnp.zeros(35),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                ),
                lambda: (
                    jnp.bool_(False),
                    make_contract_log(state, cum_return),
                ),
            )

        def make_contract_log(state, cum_return):
            actor_contract = jnp.zeros(35)
            opp_contract = jnp.zeros(35)
            actor_contract, opp_contract = jax.lax.cond(
                state._last_bidder < 2,
                lambda: (
                    actor_contract.at[state._last_bid].set(1),
                    opp_contract,
                ),
                lambda: (
                    actor_contract,
                    opp_contract.at[state._last_bid].set(1),
                ),
            )
            actor_doubled_contract, actor_redoubled_contract = jax.lax.cond(
                state._last_bidder < 2,
                lambda: (state._call_x, state._call_xx),
                lambda: (jnp.bool_(False), jnp.bool_(False)),
            )
            opp_doubled_contract, opp_redoubled_contract = jax.lax.cond(
                state._last_bidder >= 2,
                lambda: (state._call_x, state._call_xx),
                lambda: (jnp.bool_(False), jnp.bool_(False)),
            )
            (
                actor_make_contract,
                opp_make_contract,
                actor_down_contract,
                opp_down_contract,
            ) = jax.lax.cond(
                cum_return >= 0,
                lambda: jax.lax.cond(
                    state._last_bidder < 2,
                    lambda: (
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                ),
                lambda: jax.lax.cond(
                    state._last_bidder < 2,
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(True),
                        jnp.bool_(False),
                    ),
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(True),
                    ),
                ),
            )
            return (
                actor_contract,
                opp_contract,
                actor_doubled_contract,
                actor_redoubled_contract,
                opp_doubled_contract,
                opp_redoubled_contract,
                actor_make_contract,
                opp_make_contract,
                actor_down_contract,
                opp_down_contract,
            )

        pass_out, (
            actor_contract,
            opp_contract,
            actor_doubled_contract,
            actor_redoubled_contract,
            opp_doubled_contract,
            opp_redoubled_contract,
            actor_make_contract,
            opp_make_contract,
            actor_down_contract,
            opp_down_contract,
        ) = jax.vmap(make_terminated_log)(state, cum_return)
        pass_out_ratio = pass_out.sum() / num_eval_envs
        actor_doubled_count = actor_doubled_contract.sum()
        actor_doubled_ratio = actor_doubled_contract.mean()
        actor_redoubled_count = actor_redoubled_contract.sum()
        actor_redoubled_ratio = actor_redoubled_contract.mean()
        opp_doubled_count = opp_doubled_contract.sum()
        opp_doubled_ratio = opp_doubled_contract.mean()
        opp_redoubled_count = opp_redoubled_contract.sum()
        opp_redoubled_ratio = opp_redoubled_contract.mean()
        actor_declarer_count = actor_contract.sum()
        actor_declarer_ratio = actor_declarer_count / num_eval_envs
        opp_declarer_count = opp_contract.sum()
        opp_declarer_ratio = opp_declarer_count / num_eval_envs
        actor_illegal_action_probs = jax.vmap(lambda x, y: x / y)(
            actor_total_illegal_action_probs, actor_step_count
        )
        opp_illegal_action_probs = jax.vmap(lambda x, y: x / y)(
            opp_total_illegal_action_probs, opp_step_count
        )
        state = state.replace(rewards=rewards)
        log_info = (
            cum_return.mean(),
            actor_illegal_action_probs.mean(),
            opp_illegal_action_probs.mean(),
            state._step_count.mean(),
            actor_bid.mean(axis=0),
            opp_bid.mean(axis=0),
            actor_contract.mean(axis=0),
            opp_contract.mean(axis=0),
            actor_declarer_ratio,
            opp_declarer_ratio,
            actor_doubled_ratio,
            actor_redoubled_ratio,
            opp_doubled_ratio,
            opp_redoubled_ratio,
            actor_make_contract.mean(),
            opp_make_contract.mean(),
            actor_down_contract.mean(),
            opp_down_contract.mean(),
            pass_out_ratio,
        )
        return state, log_info

    def duplicate_evaluate(actor_params, rng_key):
        step_fn = duplicate_step(eval_env.step)
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)
        # state.save_svg("svg/eval_init.svg")
        cum_return = jnp.zeros(num_eval_envs)
        i = 0
        states = []
        table_a_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )
        table_b_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )

        cum_return = jnp.zeros(num_eval_envs)
        actor_total_illegal_action_probs = jnp.zeros(num_eval_envs)
        opp_total_illegal_action_probs = jnp.zeros(num_eval_envs)
        actor_step_count = jnp.zeros(num_eval_envs)
        opp_step_count = jnp.zeros(num_eval_envs)
        actor_bid = jnp.zeros((num_eval_envs, 35))
        opp_bid = jnp.zeros((num_eval_envs, 35))
        actor_pass_count = jnp.zeros(num_eval_envs)
        opp_pass_count = jnp.zeros(num_eval_envs)
        count = 0

        def cond_fn(tup):
            (state, _, _, _, _, _, _) = tup
            return ~state.terminated.all()

        def actor_make_action(state):
            logits, value = actor_forward_pass.apply(
                actor_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float64).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), pi.probs)

        def make_action(state):
            return jax.lax.cond(
                (state.current_player == 0) | (state.current_player == 1),
                lambda: actor_make_action(state),
                lambda: opp_make_action(state),
            )

        def update_log_info(pi_probs, mask, current_player, action, log_info):
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ) = log_info
            illegal_action_prob = jnp.dot(pi_probs, ~mask)
            actor_total_illegal_action_probs = jax.lax.cond(
                current_player < 2,
                lambda: actor_total_illegal_action_probs + illegal_action_prob,
                lambda: actor_total_illegal_action_probs,
            )
            opp_total_illegal_action_probs = jax.lax.cond(
                current_player >= 2,
                lambda: opp_total_illegal_action_probs + illegal_action_prob,
                lambda: opp_total_illegal_action_probs,
            )
            actor_step_count = jax.lax.cond(
                current_player < 2,
                lambda: actor_step_count + 1,
                lambda: actor_step_count,
            )
            opp_step_count = jax.lax.cond(
                current_player >= 2,
                lambda: opp_step_count + 1,
                lambda: opp_step_count,
            )
            actor_bid = jax.lax.cond(
                (current_player < 2) & (action >= 3),
                lambda: jnp.zeros(35).at[action - 3].set(1) + actor_bid,
                lambda: actor_bid,
            )
            opp_bid = jax.lax.cond(
                (current_player >= 2) & (action >= 3),
                lambda: jnp.zeros(35).at[action - 3].set(1) + opp_bid,
                lambda: opp_bid,
            )
            actor_pass_count = jax.lax.cond(
                (current_player < 2) & (action == 0),
                lambda: actor_pass_count + 1,
                lambda: actor_pass_count,
            )
            opp_pass_count = jax.lax.cond(
                (current_player >= 2) & (action == 0),
                lambda: opp_pass_count + 1,
                lambda: opp_pass_count,
            )
            return (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            )

        def make_step_log(state, pi_probs, action, log_info):
            return jax.lax.cond(
                state.terminated,
                lambda: log_info,
                lambda: update_log_info(
                    pi_probs,
                    state.legal_action_mask,
                    state.current_player,
                    action,
                    log_info,
                ),
            )

        def loop_fn(tup):
            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                log_info,
                rng_key,
                count,
            ) = tup
            (action, pi_probs) = jax.vmap(make_action)(state)

            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ) = jax.vmap(make_step_log)(state, pi_probs, action, log_info)

            rng_key, _rng = jax.random.split(rng_key)
            (state, table_a_info, table_b_info) = jax.vmap(step_fn)(
                state, action, table_a_info, table_b_info
            )
            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards, jnp.zeros_like(state.current_player)
            )
            count += 1
            # state.save_svg(f"svg/{count}.svg")
            return (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                (
                    actor_total_illegal_action_probs,
                    opp_total_illegal_action_probs,
                    actor_step_count,
                    opp_step_count,
                    actor_bid,
                    opp_bid,
                    actor_pass_count,
                    opp_pass_count,
                ),
                rng_key,
                count,
            )

        (
            state,
            table_a_info,
            table_b_info,
            cum_return,
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
                actor_pass_count,
                opp_pass_count,
            ),
            _,
            count,
        ) = jax.lax.while_loop(
            cond_fn,
            loop_fn,
            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                (
                    actor_total_illegal_action_probs,
                    opp_total_illegal_action_probs,
                    actor_step_count,
                    opp_step_count,
                    actor_bid,
                    opp_bid,
                    actor_pass_count,
                    opp_pass_count,
                ),
                rng_key,
                count,
            ),
        )
        # state.save_svg("svg/duplicate.svg")

        def make_terminated_log(table_info):
            return jax.lax.cond(
                (table_info.last_bidder == -1) & (table_info.last_bid == -1),
                lambda: (
                    jnp.bool_(True),
                    (
                        jnp.zeros(35),
                        jnp.zeros(35),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                ),
                lambda: (
                    jnp.bool_(False),
                    make_contract_log(table_info),
                ),
            )

        def make_contract_log(table_info):
            actor_contract = jnp.zeros(35)
            opp_contract = jnp.zeros(35)
            actor_contract, opp_contract = jax.lax.cond(
                table_info.last_bidder < 2,
                lambda: (
                    actor_contract.at[table_info.last_bid].set(1),
                    opp_contract,
                ),
                lambda: (
                    actor_contract,
                    opp_contract.at[table_info.last_bid].set(1),
                ),
            )
            actor_doubled_contract, actor_redoubled_contract = jax.lax.cond(
                table_info.last_bidder < 2,
                lambda: (table_info.call_x, table_info.call_xx),
                lambda: (jnp.bool_(False), jnp.bool_(False)),
            )
            opp_doubled_contract, opp_redoubled_contract = jax.lax.cond(
                table_info.last_bidder >= 2,
                lambda: (table_info.call_x, table_info.call_xx),
                lambda: (jnp.bool_(False), jnp.bool_(False)),
            )
            (
                actor_make_contract,
                opp_make_contract,
                actor_down_contract,
                opp_down_contract,
            ) = jax.lax.cond(
                table_info.rewards[0] >= 0,
                lambda: jax.lax.cond(
                    table_info.last_bidder < 2,
                    lambda: (
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    ),
                ),
                lambda: jax.lax.cond(
                    table_info.last_bidder < 2,
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(True),
                        jnp.bool_(False),
                    ),
                    lambda: (
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(False),
                        jnp.bool_(True),
                    ),
                ),
            )
            return (
                actor_contract,
                opp_contract,
                actor_doubled_contract,
                actor_redoubled_contract,
                opp_doubled_contract,
                opp_redoubled_contract,
                actor_make_contract,
                opp_make_contract,
                actor_down_contract,
                opp_down_contract,
            )

        actor_illegal_action_probs = jax.vmap(lambda x, y: x / y)(
            actor_total_illegal_action_probs, actor_step_count
        )
        opp_illegal_action_probs = jax.vmap(lambda x, y: x / y)(
            opp_total_illegal_action_probs, opp_step_count
        )

        table_a_pass_out, (
            table_a_actor_contract,
            table_a_opp_contract,
            table_a_actor_doubled_contract,
            table_a_actor_redoubled_contract,
            table_a_opp_doubled_contract,
            table_a_opp_redoubled_contract,
            table_a_actor_make_contract,
            table_a_opp_make_contract,
            table_a_actor_down_contract,
            table_a_opp_down_contract,
        ) = jax.vmap(make_terminated_log)(table_a_info)
        table_a_pass_out_ratio = table_a_pass_out.sum() / num_eval_envs
        table_a_actor_doubled_count = table_a_actor_doubled_contract.sum()
        table_a_actor_redoubled_count = table_a_actor_redoubled_contract.sum()
        table_a_opp_doubled_count = table_a_opp_doubled_contract.sum()
        table_a_opp_redoubled_count = table_a_opp_redoubled_contract.sum()
        table_a_actor_declarer_count = table_a_actor_contract.sum()
        table_a_actor_declarer_ratio = table_a_actor_declarer_count / num_eval_envs
        table_a_opp_declarer_count = table_a_opp_contract.sum()
        table_a_opp_declarer_ratio = table_a_opp_declarer_count / num_eval_envs
        table_a_score_reward = jax.vmap(get_fn)(
            table_a_info.rewards, jnp.zeros_like(state.current_player)
        )
        table_b_pass_out, (
            table_b_actor_contract,
            table_b_opp_contract,
            table_b_actor_doubled_contract,
            table_b_actor_redoubled_contract,
            table_b_opp_doubled_contract,
            table_b_opp_redoubled_contract,
            table_b_actor_make_contract,
            table_b_opp_make_contract,
            table_b_actor_down_contract,
            table_b_opp_down_contract,
        ) = jax.vmap(make_terminated_log)(table_b_info)
        table_b_pass_out_ratio = table_b_pass_out.sum() / num_eval_envs
        table_b_actor_doubled_count = table_b_actor_doubled_contract.sum()
        table_b_actor_redoubled_count = table_b_actor_redoubled_contract.sum()
        table_b_opp_doubled_count = table_b_opp_doubled_contract.sum()
        table_b_opp_redoubled_count = table_b_opp_redoubled_contract.sum()
        table_b_actor_declarer_count = table_b_actor_contract.sum()
        table_b_actor_declarer_ratio = table_b_actor_declarer_count / num_eval_envs
        table_b_opp_declarer_count = table_b_opp_contract.sum()
        table_b_opp_declarer_ratio = table_b_opp_declarer_count / num_eval_envs
        table_b_score_reward = jax.vmap(get_fn)(
            table_b_info.rewards, jnp.zeros_like(state.current_player)
        )
        std_error = jnp.std(cum_return, ddof=1) / jnp.sqrt(len(cum_return))
        log_info = (
            cum_return.mean(),
            std_error,
            (table_a_score_reward.mean() + table_b_score_reward.mean()) / 2,
            actor_illegal_action_probs.mean(),
            opp_illegal_action_probs.mean(),
            state._step_count.mean(),
            actor_bid.mean(axis=0) / 2,
            opp_bid.mean(axis=0) / 2,
            (table_a_actor_contract.mean(axis=0) + table_b_actor_contract.mean(axis=0))
            / 2,
            (table_a_opp_contract.mean(axis=0) + table_b_opp_contract.mean(axis=0)) / 2,
            (table_a_actor_declarer_ratio + table_b_actor_declarer_ratio) / 2,
            (table_a_opp_declarer_ratio + table_b_opp_declarer_ratio) / 2,
            (
                table_a_actor_doubled_contract.mean()
                + table_b_actor_doubled_contract.mean()
            )
            / 2,
            (
                table_a_actor_redoubled_contract.mean()
                + table_b_actor_redoubled_contract.mean()
            )
            / 2,
            (table_a_opp_doubled_contract.mean() + table_b_opp_doubled_contract.mean())
            / 2,
            (
                table_a_opp_redoubled_contract.mean()
                + table_b_opp_redoubled_contract.mean()
            )
            / 2,
            (table_a_actor_make_contract.mean() + table_b_actor_make_contract.mean())
            / 2,
            (table_a_opp_make_contract.mean() + table_b_opp_make_contract.mean()) / 2,
            (table_a_actor_down_contract.mean() + table_b_actor_down_contract.mean())
            / 2,
            (table_a_opp_down_contract.mean() + table_b_opp_down_contract.mean()) / 2,
            (table_a_pass_out_ratio + table_b_pass_out_ratio) / 2,
            (actor_pass_count / actor_step_count).mean(),
            (opp_pass_count / opp_step_count).mean(),
        )
        return log_info, table_a_info, table_b_info

    if duplicate:
        return duplicate_evaluate
    else:
        return evaluate


def make_evaluate_log(log_info):
    (
        IMP_return_mean,
        std_error,
        score_return_mean,
        actor_illegal_action_probs_mean,
        opp_illegal_action_probs_mean,
        step_count_mean,
        actor_bid_mean,
        opp_bid_mean,
        actor_contract_mean,
        opp_contract_mean,
        actor_declarer_ratio,
        opp_declarer_ratio,
        actor_doubled_ratio,
        actor_redoubled_ratio,
        opp_doubled_ratio,
        opp_redoubled_ratio,
        actor_make_contract_mean,
        opp_make_contract_mean,
        actor_down_contract_mean,
        opp_down_contract_mean,
        pass_out_ratio,
        actor_pass_ratio,
        opp_pass_ratio,
    ) = log_info

    log = {
        "eval/IMP_reward": IMP_return_mean,
        "eval/IMP_SE": std_error,
        "eval/score_reward": score_return_mean,
        "eval/actor_illegal_action_probs": actor_illegal_action_probs_mean,
        "eval/opp_illegal_action_probs": opp_illegal_action_probs_mean,
        "eval/step count": step_count_mean,
        "eval/actor_declarer_ratio": actor_declarer_ratio,
        "eval/opp_declarer_ratio": opp_declarer_ratio,
        "eval/actor_doubled_ratio": actor_doubled_ratio,
        "eval/actor_redoubled_ratio": actor_redoubled_ratio,
        "eval/opp_doubled_ratio": opp_doubled_ratio,
        "eval/opp_redoubled_ratio": opp_redoubled_ratio,
        "eval/actor_make_contract_ratio": actor_make_contract_mean,
        "eval/opp_make_contract_ratio": opp_make_contract_mean,
        "eval/actor_down_contract_ratio": actor_down_contract_mean,
        "eval/opp_down_contract_ratio": opp_down_contract_mean,
        "eval/pass_out_ratio": pass_out_ratio,
        "eval/actor_pass_ratio": actor_pass_ratio,
        "eval/opp_pass_ratio": opp_pass_ratio,
    }

    suits = ["C", "D", "H", "S", "NT"]
    numbers = range(1, 8)
    actor_bid_probs_dict = {}
    opp_bid_probs_dict = {}
    actor_contract_probs_dict = {}
    opp_contract_probs_dict = {}
    index = 0

    for number in numbers:
        for suit in suits:
            actor_bid_probs_dict[
                f"eval/actor_bid_probs/{number}{suit}"
            ] = actor_bid_mean[index]
            opp_bid_probs_dict[f"eval/opp_bid_probs/{number}{suit}"] = opp_bid_mean[
                index
            ]
            actor_contract_probs_dict[
                f"eval/actor_contract_probs/{number}{suit}"
            ] = actor_contract_mean[index]
            opp_contract_probs_dict[
                f"eval/opp_contract_probs/{number}{suit}"
            ] = opp_contract_mean[index]
            index += 1
    log = {
        **log,
        **actor_bid_probs_dict,
        **actor_contract_probs_dict,
        **opp_bid_probs_dict,
        **opp_contract_probs_dict,
    }

    return log


if __name__ == "__main__":
    key = "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
    wandb.login(key=key)
    config = {
        "ACTOR_ACTIVATION": "relu",
        "ACTOR_MODEL_TYPE": "FAIR",
        "OPP_ACTIVATION": "relu",
        "OPP_MODEL_TYPE": "DeepMind",
        "OPP_MODEL_PATH": "sl_log/sl_deepmind/params-400000.pkl",
        "NUM_EVAL_ENVS": 4,
        "LOG_PATH": "",
        "EXP_NAME": "",
        "PARAM_PATH": "rl_log/exp0067/rl_params/params-00000085.pkl",
        "TRACK": True,
        "GAME_MODE": "competitive",
    }
    if config["TRACK"]:
        wandb.init(project="eval_test", config=config)
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    env = BridgeBidding()
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = actor_forward_pass.init(_rng, init_x)
    params = pickle.load(
        open(
            os.path.join(
                config["LOG_PATH"],
                config["EXP_NAME"],
                config["PARAM_PATH"],
            ),
            "rb",
        )
    )
    evaluate = make_evaluate(config)
    duplicate_evaluate = make_evaluate(config, duplicate=True)
    jit_evaluate = jax.jit(evaluate)
    print("start")
    # state, log_info = jit_evaluate(params, rng)
    """
    if config["TRACK"]:
        state.save_svg(
            os.path.join(config["LOG_PATH"], config["EXP_NAME"], "vs_sl.svg")
        )
    """
    # log = make_evaluate_log(log_info)
    # pprint(log)
    log_info, table_a_info, table_b_info = jax.jit(duplicate_evaluate)(params, rng)
    print(make_evaluate_log(log_info))
    print(table_a_info)
    print(table_b_info)
    # state.save_svg("svg/test_du.svg")
    # state.save_svg("svg/test.svg")
    # print(state)
    log = jax.jit(make_evaluate_log)(log_info)
    pprint(log)

    if config["TRACK"]:
        wandb.log(log)
