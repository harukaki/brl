"""
This code is based on https://github.com/luchris429/purejaxrl
"""
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
    single_play_step_two_policy_commpetitive
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


class PPOConfig(BaseModel):
    ENV_NAME: Literal["bridge_bidding"] = "bridge_bidding"
    LR: float = 0.000001  # 0.0003
    #    EVAL_ENVS: int = 100
    NUM_ENVS: int = 4096  # 並列pgx環境数　evalで計算されるゲーム数
    NUM_STEPS: int = 64
    TOTAL_TIMESTEPS: int = 200000000
    UPDATE_EPOCHS: int = 5  # 一回のupdateでbatchが何回学習されるか
    NUM_MINIBATCHES: int = 128
    GAMMA: float = 1
    GAE_LAMBDA: float = 0.95
    # GAMMA: float = 0.99
    # GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    # ENT_COEF: float = 0.01
    ENT_COEF: float = 0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTOR_ACTIVATION: str = "relu"
    ACTOR_MODEL_TYPE: Literal["DeepMind", "FAIR"] = "FAIR"
    OPP_ACTIVATION: str = "relu"
    OPP_MODEL_TYPE: Literal["DeepMind", "FAIR"] = "DeepMind"
    NUM_UPDATES: int = (
        10000  # updateが何回されるか　TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENV
    )
    MINIBATCH_SIZE: int = 1024  # update中の1 epochで使用される数
    ANNEAL_LR: bool = False  # True
    VS_RANDOM: bool = False
    UPDATE_INTERVAL: int = 5
    MAKE_ANCHOR: bool = True
    REWARD_SCALE: float = 7600
    NUM_EVAL_ENVS: int = 1000
    DDS_RESULTS_DIR: str = "dds_results"
    HASH_SIZE: int = 100_000
    TRAIN_SIZE: int = 2_500_000
    LOAD_INITIAL_MODEL: bool = False



def make_forward_pass(activation, model_type):
    def forward_fn(x):
        net = ActorCritic(
            38,
            activation=activation,
            model=model_type,
        )
        logits, value = net(x)
        return logits, value

    return hk.without_apply_rng(hk.transform(forward_fn))


def actor_forward_fn(x):
    net = ActorCritic(
        38,
        activation="relu",
    )
    logits, value = net(x)
    return logits, value


def opp_forward_fn(x):
    net = ActorCritic(38, activation="relu")
    logits, value = net(x)
    return logits, value


rl_forward_pass = hk.without_apply_rng(hk.transform(actor_forward_fn))
sl_forward_pass = hk.without_apply_rng(hk.transform(DeepMind_sl_net_fn))
DeepMind_sl_model_param = pickle.load(
    open("bridge_bidding_sl_networks/params-290000.pkl", "rb")
)


def wraped_sl_forward_pass_apply(observation):
    return sl_forward_pass.apply(DeepMind_sl_model_param, observation)


def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
if args.ANNEAL_LR:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.MAX_GRAD_NORM),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
else:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.MAX_GRAD_NORM),
        optax.adam(args.LR, eps=1e-5),
    )


def _make_step(env_name, params, eval=False):
    env = bb.BridgeBidding(
        dds_hash_table_path="dds_hash_table/dds_hash_table_0000k.npy"
    )
    step_fn = auto_reset(env.step, env.init) if not eval else env.step
    return single_play_step_vs_policy_in_bridge_bidding(
        step_fn, rl_forward_pass, params
    )


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def make_update_fn(config, env_step_fn, env_init_fn):
    actor_forward_pass = make_forward_pass(activation=config["ACTOR_ACCTIVATINO"], model_type=config["ACTOR_MODEL_TYPE"])
    opp_forward_pass = make_forward_pass(activation=config["OPP_ACCTIVATINO"], model_type=config["OPP_MODEL_TYPE"])


    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES

        # step_fn = _make_step(config["ENV_NAME"], runner_state[0])  # DONE
        """
        step_fn = single_play_step_vs_DeepMind_sl_model(
            step_fn=auto_reset(env_step_fn, env_init_fn),
            actor_forward_pass=rl_forward_pass,
            actor_params=runner_state[0],
            sl_forward_pass_apply=wraped_sl_forward_pass_apply,
        )
        get_fn = _get
        """
        step_fn = single_play_step_two_policy_commpetitive(
            step_fn=auto_reset(env_step_fn, env_init_fn),
            actor_forward_pass=rl_forward_pass,
            actor_params=runner_state[0],
            opp_forward_pass=,
        )

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
            # SELECT ACTION
            # print("===00==")
            # print(f"current player: {env_state.current_player}")
            rng, _rng = jax.random.split(rng)
            logits, value = rl_forward_pass.apply(
                params,
                last_obs.astype(jnp.float32),
            )  # DONE
            mask = env_state.legal_action_mask
            logits = logits + jnp.finfo(np.float64).min * (~mask)
            pi = distrax.Categorical(logits=logits)
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
                jax.vmap(get_fn)(
                    env_state.rewards / config["REWARD_SCALE"], actor
                ),
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

        # CALCULATE ADVANTAGE
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state  # DONE
        _, last_val = rl_forward_pass.apply(
            params, last_obs.astype(jnp.float32)
        )  # DONE

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = (
                    reward + config["GAMMA"] * next_value * (1 - done) - value
                )
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            """
            def scan(f, init, xs, length=None):
                if xs is None:
                    xs = [None] * length
                carry = init
                ys = []
                for x in xs:
                    carry, y = f(carry, x)
                    ys.append(y)
                return carry, ys

            _, advantages = scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
            )
            """

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # print(traj_batch)
        # print(last_val)

        advantages, targets = _calculate_gae(traj_batch, last_val)
        # print(advantages)
        # print(targets)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = rl_forward_pass.apply(
                        params, traj_batch.obs.astype(jnp.float32)
                    )  # DONE
                    mask = traj_batch.legal_action_mask
                    logits = logits + jnp.finfo(np.float64).min * (~mask)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - targets
                    )
                    value_loss = (
                        0.5
                        * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)

                    # gae標準化
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    params, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)  # DONE
                return (
                    params,
                    opt_state,
                ), total_loss  # DONE

            (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state  # DONE
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )  # DONE
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )  # DONE
            return update_state, total_loss

        update_state = (
            params,
            opt_state,
            traj_batch,
            advantages,
            targets,
            rng,
        )  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        params, opt_state, _, _, _, rng = update_state  # DONE

        runner_state = (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        )  # DONE
        return runner_state, loss_info

    return _update_step


def _get(x, i):
    return x[i]


def _get_zero(x, i):
    return x[0]


eval_env = bb.BridgeBidding("dds_results/test_000.npy")


def evaluate(params, rng_key):
    num_eval_envs = 1000
    step_fn = single_play_step_vs_DeepMind_sl_model_in_deterministic(
        step_fn=eval_env.step,
        actor_forward_pass=rl_forward_pass,
        actor_params=params,
        sl_forward_pass_apply=wraped_sl_forward_pass_apply,
    )
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_eval_envs)
    state = jax.vmap(eval_env.init)(subkeys)
    # state.save_svg("svg/eval_init.svg")
    cum_return = jnp.zeros(num_eval_envs)
    get_fn = _get

    def cond_fn(tup):
        (
            state,
            _,
            _,
        ) = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, cum_return, rng_key = tup
        actor = state.current_player
        logits, value = rl_forward_pass.apply(
            params, state.observation.astype(jnp.float32)
        )  # DONE
        logits = logits + jnp.finfo(np.float64).min * (
            ~state.legal_action_mask
        )
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        # determinestic
        action = pi.mode()
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action, _rng)
        cum_return = cum_return + jax.vmap(get_fn)(state.rewards, actor) / 7600
        return state, cum_return, rng_key

    state, cum_return, _ = jax.lax.while_loop(
        cond_fn,
        loop_fn,
        (state, cum_return, rng_key),
    )

    return cum_return.mean()


def duplicate_evaluate(params, rng_key):
    num_eval_envs = 1000
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

    def rl_make_action(state, actor_params):
        logits_old, value = rl_forward_pass.apply(
            params, state.observation
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

    def sl_make_action(state, opp_params):
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

    def make_action(state):
        return jax.lax.cond(
            (state.current_player == 0) | (state.current_player == 1),
            lambda: rl_make_action(state),
            lambda: sl_make_action(state),
        )

    def loop_fn(tup):
        state, table_a_reward, has_duplicate_result, cum_return, rng_key = tup
        action, probs, logits, mask, logits_old, mask_logits = jax.vmap(
            make_action
        )(state)
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

    return cum_return.mean()


def train(config, rng):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    )

    env = bb.BridgeBidding()
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = rl_forward_pass.init(_rng, init_x)  # params  # DONE
    opt_state = optimizer.init(params=params)  # DONE

    jit_evaluate = jax.jit(evaluate)
    jit_diplicate_evaluate = jax.jit(duplicate_evaluate)
    # jit_duplicate_evaluate = jax.jit(duplicate_evaluate)
    # INIT UPDATE FUNCTION
    _update_step = make_update_fn(config, env.step, env.init)  # DONE
    jitted_update_step = jax.jit(_update_step)
    jitted_init = jax.jit(jax.vmap(env.init))

    # INIT ENV
    compile_sta = time.time()
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    compile_end = time.time()

    print(f"compile time: {compile_end - compile_sta}")

    env_state = jitted_init(reset_rng)
    terminated_count = 0
    rng, _rng = jax.random.split(rng)
    runner_state = (
        params,
        opt_state,
        env_state,
        env_state.observation,
        terminated_count,
        _rng,
    )  # DONE

    anchor_model = None
    ckpt_filename = f'checkpoints/{config["ENV_NAME"]}/model.ckpt'
    anchor_filename = f'checkpoints/{config["ENV_NAME"]}/anchor.ckpt'
    if anchor_filename != "" and os.path.isfile(anchor_filename):
        with open(ckpt_filename, "rb") as reader:
            dic = pickle.load(reader)
        anchor_model = dic["params"]

    steps = 0
    train_dds_results_list = sorted(
        [
            path
            for path in os.listdir(config["DDS_RESULTS_DIR"])
            if "train" in path
        ]
    )
    hash_index = 0
    board_count = 0
    for i in range(config["NUM_UPDATES"]):
        time_eval_sta = time.time()
        if anchor_model is not None and not config["MAKE_ANCHOR"]:
            """step_fn = _make_step(
                config["ENV_NAME"], anchor_model, eval=True
            )  # DONE"""
            eval_R = jit_evaluate(runner_state[0], rng)  # DONE
            du_eval_R = jit_diplicate_evaluate(runner_state[0], rng)
        else:
            """step_fn = _make_step(
                config["ENV_NAME"], runner_state[0], eval=True
            )  # DONE"""
            eval_R = jit_evaluate(runner_state[0], rng)  # DONE
            du_eval_R = jit_diplicate_evaluate(runner_state[0], rng)
        time_eval_end = time.time()
        print(f"eval time: {time_eval_end-time_eval_sta}")
        # print(log)
        # wandb.log(log)
        time1 = time.time()
        runner_state, loss_info = jitted_update_step(runner_state)  # DONE
        # print(f"boad_num: {runner_state[4]}")
        # runner_state, loss_info = _update_step(runner_state)
        time2 = time.time()
        print(f"update time: {time2 - time1}")
        steps += config["NUM_ENVS"] * config["NUM_STEPS"]
        if config["MAKE_ANCHOR"]:
            with open(
                f"checkpoints/{config['ENV_NAME']}/anchor.ckpt", "wb"
            ) as writer:
                pickle.dump(
                    {"params": runner_state[0], "opt_state": runner_state[1]},
                    writer,
                )
        if i % 10 == 0:
            with open(
                f"checkpoints/{config['ENV_NAME']}/model.ckpt", "wb"
            ) as writer:
                pickle.dump(
                    {"params": runner_state[0], "opt_state": runner_state[1]},
                    writer,
                )
        total_loss, (value_loss, loss_actor, entropy) = loss_info

        # make log
        log = {
            "train/total_loss": float(total_loss.mean().mean()),
            "train/value_loss": float(value_loss.mean().mean()),
            "train/loss_actor": float(loss_actor.mean().mean()),
            "train/policy_entropy": float(entropy.mean().mean()),
            "eval/score_reward": float(eval_R),
            "eval/IMP_reward": float(du_eval_R),
            "board_num": int(runner_state[4]),
            "steps": steps,
        }
        pprint(log)
        wandb.log(log)
        if (runner_state[4] - board_count) // config["HASH_SIZE"] >= 1:
            hash_index += 1
            print(f"board count: {runner_state[4] - board_count}")
            board_count = runner_state[4]
            if hash_index == len(train_dds_results_list):
                hash_index = 0
                random.shuffle(train_dds_results_list)
            hash_path = os.path.join(
                config["DDS_RESULTS_DIR"], train_dds_results_list[hash_index]
            )
            reload_sta = time.time()
            env = bb.BridgeBidding(hash_path)
            _update_step = make_update_fn(config, env.step, env.init)  # DONE
            jitted_update_step = jax.jit(_update_step)
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            jitted_init = jax.jit(jax.vmap(env.init))
            env_state = jitted_init(reset_rng)
            reload_end = time.time()
            print(f"reload time: {reload_end - reload_sta}")
            runner_state = (
                runner_state[0],
                runner_state[1],
                env_state,
                env_state.observation,
                runner_state[4],
                _rng,
            )
    return runner_state


if __name__ == "__main__":
    key = "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
    wandb.login(key=key)
    mode = "make-anchor" if args.MAKE_ANCHOR else "train"
    wandb.init(project=f"ppo-{mode}-bridge", config=args.dict())
    config = {
        "LR": args.LR,
        "NUM_ENVS": args.NUM_ENVS,
        "NUM_STEPS": args.NUM_STEPS,
        "TOTAL_TIMESTEPS": args.TOTAL_TIMESTEPS,
        "UPDATE_EPOCHS": args.UPDATE_EPOCHS,
        "MINIBATCH_SIZE": args.MINIBATCH_SIZE,
        "GAMMA": args.GAMMA,
        "GAE_LAMBDA": args.GAE_LAMBDA,
        "CLIP_EPS": args.CLIP_EPS,
        "ENT_COEF": args.ENT_COEF,
        "VF_COEF": args.VF_COEF,
        "MAX_GRAD_NORM": args.MAX_GRAD_NORM,
        "ACTIVATION": args.ACTIVATION,
        "ENV_NAME": args.ENV_NAME,
        "ANNEAL_LR": True,
        "VS_RANDOM": args.VS_RANDOM,
        "UPDATE_INTERVAL": args.UPDATE_INTERVAL,
        "MAKE_ANCHOR": args.MAKE_ANCHOR,
        "NUM_EVAL_ENVS": args.NUM_EVAL_ENVS,
        "DDS_RESULTS_DIR": args.DDS_RESULTS_DIR,
        "HASH_SIZE": args.HASH_SIZE,
        "REWARD_SCALE": args.REWARD_SCALE,
        "NUM_EVAL_ENVS": args.NUM_EVAL_ENVS
    }
    print("training of", config["ENV_NAME"])
    rng = jax.random.PRNGKey(0)
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
