import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from typing import Sequence, NamedTuple, Any, Literal, Type
import distrax
import os
import pgx.bridge_bidding as bb


import pickle
import wandb
from duplicate import duplicate_step, Table_info
from models import make_forward_pass
from pprint import pprint


"""
def make_act(act_type):
    if act_type == "param_act":
        forward_pass = make_forward_pass()

        def param_act(state, forward_pass, params):
            logits_old, value = forward_pass.apply(
                params, state.observation
            )  # DONE
            mask_logits = jnp.finfo(np.float32).min * (
                ~state.legal_action_mask
            )
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

        return param_act
    elif act_type == "pass_act":

        def pass_act(state):
            pi_probs = jnp.zeros(38).at[0].set(True)
            return (
                jnp.int32(0),
                pi_probs,
                jnp.float32(1),
                state.legal_action_mask,
                jnp.float32(1),
                jnp.float32(1),
            )

        return pass_act
"""


def make_evaluate(config, duplicate=False):
    eval_env = bb.BridgeBidding("dds_results/test_000.npy")
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    opp_forward_pass = make_forward_pass(
        activation=config["OPP_ACTIVATION"],
        model_type=config["OPP_MODEL_TYPE"],
    )
    opp_params = pickle.load(open(config["OPP_MODEL_PATH"], "rb"))
    num_eval_envs = config["NUM_EVAL_ENVS"]

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

        def cond_fn(tup):
            (
                state,
                _,
                _,
            ) = tup
            return ~state.terminated.all()

        def rl_make_action(state):
            logits_old, value = actor_forward_pass.apply(
                actor_params, state.observation
            )  # DONE
            mask_logits = jnp.finfo(np.float32).min * (
                ~state.legal_action_mask
            )
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
            mask_logits = jnp.finfo(np.float32).min * (
                ~state.legal_action_mask
            )
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

        def update_log_info(
            logits_old, mask, current_player, action, log_info
        ):
            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
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
            return (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
            )

        def make_step_log(
            state, logits_old, mask, current_player, action, log_info
        ):
            return jax.lax.cond(
                state.terminated,
                lambda: log_info,
                lambda: update_log_info(
                    logits_old, mask, current_player, action, log_info
                ),
            )

        def loop_fn(tup):
            (
                state,
                cum_return,
                log_info,
            ) = tup

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
            ) = jax.vmap(make_step_log)(
                state, logits_old, mask, state.current_player, action, log_info
            )
            """
            print(f"terminated: {state.terminated}")
            print(f"current player: {state.current_player}")
            print(f"action prob: {probs}")
            print(f"illegal action: {state.legal_action_mask}")
            print(f"action: {action}")
            print(f"act illegal action: {actor_total_illegal_action_probs}")
            print(f"opp illegal action: {opp_total_illegal_action_probs}")
            print(f"act step: {actor_step_count}")
            print(f"opp step: {opp_step_count}")
            print(f"act bid: {actor_bid}")
            print(f"opp bid: {opp_bid}")
            print(state)
            state.save_svg("svg/test.svg")
            """
            state = jax.vmap(step_fn)(state, action)

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
                ),
            )

        """
        def while_loop(cond_fun, body_fun, init_val):
            val = init_val
            while cond_fun(val):
                val = body_fun(val)
            return val
        """

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
            ),
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
                ),
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

        """
        print(state)
        state.save_svg("svg/test_eval.svg")
        print(f"cum_return: {cum_return}")
        print(f"actor_total_illegal_action_probs: {actor_total_illegal_action_probs}")
        print(f"opp_total_illegal_action_probs: {opp_total_illegal_action_probs}")
        print(f"actor_illegal_action_probs: {actor_illegal_action_probs}")
        print(f"opp_illegal_action_probs: {opp_illegal_action_probs}")
        print(f"actor_bid: {actor_bid}")
        print(f"actor_bid.mean: {actor_bid.mean(axis=0)}")
        print(f"actor_contract: {actor_contract}")
        print(f"actor_contract.mean: {actor_contract.mean(axis=0)}")
        print(f"opp_bid: {opp_bid}")
        print(f"opp_bid.mean: {opp_bid.mean(axis=0)}")
        print(f"opp_contract: {opp_contract}")
        print(f"opp_contract.mean: {opp_contract.mean(axis=0)}")
        print(f"actor_declarer_count: {actor_declarer_count}")
        print(f"actor_declarer_ratio: {actor_declarer_ratio}")
        print(f"opp_declarer_count: {opp_declarer_count}")
        print(f"opp_declarer_ratio: {opp_declarer_ratio}")
        print(f"actor_doubled_count: {actor_doubled_count}")
        print(f"actor_doubled_ratio: {actor_doubled_ratio}")
        print(f"actor_redoubled_count: {actor_redoubled_count}")
        print(f"actor_redoubled_ratio: {actor_redoubled_ratio}")
        print(f"opp_doubled_count: {opp_doubled_count}")
        print(f"opp_doubled_ratio: {opp_doubled_ratio}")
        print(f"opp_redoubled_count: {opp_redoubled_count}")
        print(f"opp_redoubled_ratio: {opp_redoubled_ratio}")
        print(f"actor_make_contract_ratio: {actor_make_contract.mean()}")
        print(f"opp_make_contract_ratio: {opp_make_contract.mean()}")
        print(f"actor_down_contract_ratio: {actor_down_contract.mean()}")
        print(f"opp_down_contract_ratio: {opp_down_contract.mean()}")
        """
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

        def cond_fn(tup):
            (
                state,
                _,
                _,
                _,
                _,
                _,
            ) = tup
            return ~state.terminated.all()

        def actor_make_action(state):
            logits, value = actor_forward_pass.apply(
                actor_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float32).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), masked_pi.probs, logits, pi.probs)

        def opp_make_action(state):
            logits, value = actor_forward_pass.apply(
                actor_params, state.observation
            )  # DONE
            masked_logits = logits + jnp.finfo(np.float32).min * (
                ~state.legal_action_mask
            )
            masked_pi = distrax.Categorical(logits=masked_logits)
            pi = distrax.Categorical(logits=logits)
            return (masked_pi.mode(), masked_pi.probs, logits, pi.probs)

        def pass_act(state):
            pi_probs = jnp.zeros(38).at[0].set(True)
            return (
                jnp.int32(0),
                pi_probs,
                jnp.zeros(38, dtype=jnp.float32),
                state.legal_action_mask,
                jnp.zeros(38, dtype=jnp.float32),
                jnp.zeros(38, dtype=jnp.float32),
            )

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
            return (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
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
            ) = tup
            (action, masked_pi_probs, logits, pi_probs) = jax.vmap(
                make_action
            )(state)

            (
                actor_total_illegal_action_probs,
                opp_total_illegal_action_probs,
                actor_step_count,
                opp_step_count,
                actor_bid,
                opp_bid,
            ) = jax.vmap(make_step_log)(state, pi_probs, action, log_info)
            print(f"terminated: {state.terminated}")
            print(f"current player: {state.current_player}")
            # print(f"masked action prob: {masked_pi_probs}")
            print(f"action prob: {pi_probs}")
            print(f"illegal action: {~state.legal_action_mask}")
            print(f"illegal action probs: {jax.vmap(jnp.dot)(pi_probs, ~state.legal_action_mask)}")
            print(f"action: {action}")
            print(f"act illegal action: {actor_total_illegal_action_probs}")
            print(f"opp illegal action: {opp_total_illegal_action_probs}")
            print(f"act step: {actor_step_count}")
            print(f"opp step: {opp_step_count}")
            print(f"act bid: {actor_bid}")
            print(f"opp bid: {opp_bid}")
            print(state._bidding_history)
            # state.save_svg("svg/test.svg")
            rng_key, _rng = jax.random.split(rng_key)
            (state, table_a_info, table_b_info) = jax.vmap(step_fn)(
                state, action, table_a_info, table_b_info
            )
            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards, jnp.zeros_like(state.current_player)
            )
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
                ),
                rng_key,
            )

        """
        (
            state,
            table_a_info,
            table_b_info,
            cum_return,
            _,
        ) = jax.lax.while_loop(
            cond_fn,
            loop_fn,
            (state, table_a_info, table_b_info, cum_return, rng_key),
        )

        """

        def while_loop(cond_fun, body_fun, init_val):
            val = init_val
            while cond_fun(val):
                val = body_fun(val)
                # print(val[0]
            return val

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
            ),
            _,
        ) = while_loop(
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
                ),
                rng_key,
            ),
        )
        # state.save_svg("svg/duplicate.svg")

        return cum_return, table_a_info, table_b_info

    if duplicate:
        return duplicate_evaluate
    else:
        return evaluate


def make_evaluate_log(log_info):
    (
        cum_return_mean,
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
    ) = log_info
    log = {
        "eval/score_reward": float(cum_return_mean),
        "eval/actor_illegal_action_probs": float(
            actor_illegal_action_probs_mean
        ),
        "eval/opp_illegal_action_probs": float(opp_illegal_action_probs_mean),
        "eval/step count": float(step_count_mean),
        "eval/actor_declarer_ratio": float(actor_declarer_ratio),
        "eval/opp_declarer_ratio": float(opp_declarer_ratio),
        "eval/actor_doubled_ratio": float(actor_doubled_ratio),
        "eval/actor_redoubled_ratio": float(actor_redoubled_ratio),
        "eval/opp_doubled_ratio": float(opp_doubled_ratio),
        "eval/opp_redoubled_ratio": float(opp_redoubled_ratio),
        "eval/actor_make_contract_ratio": float(actor_make_contract_mean),
        "eval/opp_make_contract_ratio": float(opp_make_contract_mean),
        "eval/actor_down_contract_ratio": float(actor_down_contract_mean),
        "eval/opp_down_contract_ratio": float(opp_down_contract_mean),
        "eval/pass_out_ratio": float(pass_out_ratio),
    }
    suits = ["C", "D", "H", "S", "NT"]
    numbers = range(1, 8)
    actor_bid_probs_keys = [
        f"eval/actor_bid_probs/{number}{suit}"
        for number in numbers
        for suit in suits
    ]
    actor_contract_probs_keys = [
        f"eval/actor_contract_probs/{number}{suit}"
        for number in numbers
        for suit in suits
    ]
    opp_bid_probs_keys = [
        f"eval/opp_bid_probs/{number}{suit}"
        for number in numbers
        for suit in suits
    ]
    opp_contract_probs_keys = [
        f"eval/opp_contract_probs/{number}{suit}"
        for number in numbers
        for suit in suits
    ]
    actor_bid_probs_dict = dict(
        zip(actor_bid_probs_keys, actor_bid_mean.astype(float))
    )
    actor_contract_probs_dict = dict(
        zip(actor_contract_probs_keys, actor_contract_mean.astype(float))
    )
    opp_bid_probs_dict = dict(
        zip(opp_bid_probs_keys, opp_bid_mean.astype(float))
    )
    opp_contract_probs_dict = dict(
        zip(opp_contract_probs_keys, opp_contract_mean.astype(float))
    )
    actor_bid_probs_dict = {
        key: float(value) for key, value in actor_bid_probs_dict.items()
    }
    actor_contract_probs_dict = {
        key: float(value) for key, value in actor_contract_probs_dict.items()
    }
    opp_bid_probs_dict = {
        key: float(value) for key, value in opp_bid_probs_dict.items()
    }
    opp_contract_probs_dict = {
        key: float(value) for key, value in opp_contract_probs_dict.items()
    }
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
        "ACTOR_MODEL_TYPE": "DeepMind",
        "OPP_ACTIVATION": "relu",
        "OPP_MODEL_TYPE": "DeepMind",
        "OPP_MODEL_PATH": "sl_params/params-300000.pkl",
        "NUM_EVAL_ENVS": 2,
        "LOG_PATH": "",
        "EXP_NAME": "",
        "PARAM_PATH": "sl_params/params-300000.pkl",
        "TRACK": False,
    }
    if config["TRACK"]:
        wandb.init(project="eval_test", config=config)
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    env = bb.BridgeBidding()
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
    du_eval_R, table_a_info, table_b_info = duplicate_evaluate(params, rng)
    print(table_a_info)
    print(table_b_info)
    print(du_eval_R)
    print(du_eval_R.mean())
    # state.save_svg("svg/test_du.svg")
    # state.save_svg("svg/test.svg")
    # print(state)

    if config["TRACK"]:
        wandb.log(log)
