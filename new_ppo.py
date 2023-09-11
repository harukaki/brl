"""
This code is based on https://github.com/luchris429/purejaxrl
"""
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from typing import NamedTuple, Any, Literal
import distrax
import pgx.bridge_bidding as bb
from src.utils import (
    auto_reset,
    single_play_step_two_policy_commpetitive,
    single_play_step_two_policy_commpetitive_deterministic,
    single_play_step_free_run,
    entropy_from_dif,
)
import time
import os
import random
import json
from pprint import pprint


import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb

from src.duplicate import duplicate_step
from src.models import ActorCritic, make_forward_pass
from src.evaluation import make_evaluate, make_evaluate_log

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
    OPP_MODEL_PATH: str = "sl_params/params-300000.pkl"
    NUM_UPDATES: int = 10000  # updateが何回されるか　TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENV
    MINIBATCH_SIZE: int = 1024  # update中の1 epochで使用される数
    ANNEAL_LR: bool = False  # True
    VS_RANDOM: bool = False
    UPDATE_INTERVAL: int = 5
    MAKE_ANCHOR: bool = True
    REWARD_SCALE: float = 7600
    NUM_EVAL_ENVS: int = 10000
    DDS_RESULTS_DIR: str = "dds_results"
    HASH_SIZE: int = 100_000
    TRAIN_SIZE: int = 12_500_000
    LOAD_INITIAL_MODEL: bool = False
    INITIAL_MODEL_PATH: str = "sl_fair_0_1/params-400000.pkl"
    SAVE_MODEL: bool = True
    LOG_PATH: str = "log"
    EXP_NAME: str = "exp_0000"
    MODEL_SAVE_PATH: str = "rl_params"
    TRACK: bool = True
    ACTOR_ILLEGAL_ACTION_MASK: bool = True
    ACTOR_ILLEGAL_ACTION_PENALTY: bool = False
    ILLEGAL_ACTION_PENALTY: float = -1
    ILLEGAL_ACTION_L2NORM_COEF: float = 0
    GAME_MODE: Literal["competitive", "free-run"] = "competitive"


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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def make_update_fn(config, env_step_fn, env_init_fn):
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )

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

    policy = make_policy(config)

    if config["GAME_MODE"] == "competitive":
        make_step_fn = single_play_step_two_policy_commpetitive
        opp_forward_pass = make_forward_pass(
            activation=config["OPP_ACTIVATION"],
            model_type=config["OPP_MODEL_TYPE"],
        )
        opp_params = pickle.load(open(config["OPP_MODEL_PATH"], "rb"))
    elif config["GAME_MODE"] == "free-run":
        make_step_fn = single_play_step_free_run
        opp_forward_pass = None
        opp_params = None

    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES

        # step_fn = _make_step(config["ENV_NAME"], runner_state[0])  # DONE
        step_fn = make_step_fn(
            step_fn=auto_reset(env_step_fn, env_init_fn),
            actor_forward_pass=actor_forward_pass,
            actor_params=runner_state[0],
            opp_forward_pass=opp_forward_pass,
            opp_params=opp_params,
        )
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
        # print(traj_batch)

        # CALCULATE ADVANTAGE
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state  # DONE
        _, last_val = actor_forward_pass.apply(
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
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                    logits, value = actor_forward_pass.apply(
                        params, traj_batch.obs.astype(jnp.float32)
                    )  # DONE
                    mask = traj_batch.legal_action_mask
                    pi = policy(mask, logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
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

                    """
                    illegal_action_masked_logits = logits + jnp.finfo(
                        np.float64
                    ).min * (~mask)
                    illegal_action_masked_pi = distrax.Categorical(
                        logits=illegal_action_masked_logits
                    )
                    entropy = illegal_action_masked_pi.entropy().mean()
                    """
                    entropy = jax.vmap(entropy_from_dif)(logits, mask).mean()

                    pi = distrax.Categorical(logits=logits)
                    illegal_action_probabilities = pi.probs * ~mask
                    illegal_action_loss = (
                        jnp.linalg.norm(illegal_action_probabilities, ord=2) / 2
                    )
                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                        + config["ILLEGAL_ACTION_L2NORM_COEF"] * illegal_action_loss
                    )

                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipflacs = jnp.float32(
                        jnp.abs((ratio - 1.0)) > config["CLIP_EPS"]
                    ).mean()

                    return total_loss, (
                        value_loss,
                        loss_actor,
                        entropy,
                        approx_kl,
                        clipflacs,
                        illegal_action_loss,
                    )

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
        # print(loss_info)
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


def train(config, rng):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_MINIBATCHES"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["MINIBATCH_SIZE"]
    )
    env = bb.BridgeBidding()

    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = actor_forward_pass.init(_rng, init_x)  # params  # DONE
    opt_state = optimizer.init(params=params)  # DONE

    # LOAD INITIAL MODEL
    if config["LOAD_INITIAL_MODEL"]:
        params = pickle.load(open(config["INITIAL_MODEL_PATH"], "rb"))
        print(f"load initial params for actor: {config['INITIAL_MODEL_PATH']}")

    # MAKE EVAL
    rng, eval_rng = jax.random.split(rng)
    evaluate = make_evaluate(config)
    duplicate_evaluate = make_evaluate(config, duplicate=True)
    jit_evaluate = jax.jit(evaluate)
    jit_diplicate_evaluate = jax.jit(duplicate_evaluate)
    jit_make_evaluate_log = jax.jit(make_evaluate_log)

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
        [path for path in os.listdir(config["DDS_RESULTS_DIR"]) if "train" in path]
    )
    hash_index = 0
    board_count = 0

    if config["SAVE_MODEL"]:
        os.mkdir(
            os.path.join(
                config["LOG_PATH"],
                config["EXP_NAME"],
                config["MODEL_SAVE_PATH"],
            )
        )

    for i in range(config["NUM_UPDATES"]):
        # eval
        time_eval_sta = time.time()
        # state, log_info = jit_evaluate(runner_state[0], eval_rng)  # DONE
        log_info, _, _ = jit_diplicate_evaluate(runner_state[0], eval_rng)
        time_eval_end = time.time()
        print(f"eval time: {time_eval_end-time_eval_sta}")
        time1 = time.time()
        runner_state, loss_info = jitted_update_step(runner_state)  # DONE
        # runner_state, loss_info = _update_step(runner_state)
        time2 = time.time()
        print(f"update time: {time2 - time1}")
        steps += config["NUM_ENVS"] * config["NUM_STEPS"]

        if config["SAVE_MODEL"]:
            with open(
                os.path.join(
                    config["LOG_PATH"],
                    config["EXP_NAME"],
                    config["MODEL_SAVE_PATH"],
                    f"params-{i:08}.pkl",
                ),
                "wb",
            ) as writer:
                pickle.dump(runner_state[0], writer)
            with open(
                os.path.join(
                    config["LOG_PATH"],
                    config["EXP_NAME"],
                    config["MODEL_SAVE_PATH"],
                    f"opt_state-{i:08}.pkl",
                ),
                "wb",
            ) as writer:
                pickle.dump(runner_state[1], writer)
        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            approx_kl,
            clipflacs,
            illegal_action_loss,
        ) = loss_info
        time_make_log_sta = time.time()
        eval_log = jit_make_evaluate_log(log_info)
        time_make_log_end = time.time()
        print(f"make log time: {time_make_log_end -time_make_log_sta}")
        """
        print(value_loss)
        print(value_loss.shape)
        print(loss_actor)
        print(loss_actor.shape)
        print(entropy)
        print(entropy.shape)
        print(approx_kl)
        print(approx_kl.shape)
        print(clipflacs)
        print(clipflacs.shape)
        """

        # make log
        log = {
            "train/total_loss": float(total_loss[-1][-1]),
            "train/value_loss": float(value_loss[-1][-1]),
            "train/loss_actor": float(loss_actor[-1][-1]),
            "train/illegal_action_loss": float(illegal_action_loss[-1][-1]),
            "train/policy_entropy": float(entropy[-1][-1]),
            "train/clipflacs": float(clipflacs[-1][-1]),
            "train/approx_kl": float(approx_kl[-1][-1]),
            "board_num": int(runner_state[4]),
            "steps": steps,
        }
        total_log = {**log, **eval_log}
        pprint(log)
        if config["TRACK"]:
            wandb.log(total_log)

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
            jitted_init = jax.jit(jax.vmap(env.init))
            jitted_update_step = jax.jit(_update_step)

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

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
    # mode = "make-anchor" if args.MAKE_ANCHOR else "train"
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
        # "ACTIVATION": args.ACTIVATION,
        "ENV_NAME": args.ENV_NAME,
        "ANNEAL_LR": True,
        "VS_RANDOM": args.VS_RANDOM,
        "UPDATE_INTERVAL": args.UPDATE_INTERVAL,
        "MAKE_ANCHOR": args.MAKE_ANCHOR,
        "NUM_EVAL_ENVS": args.NUM_EVAL_ENVS,
        "DDS_RESULTS_DIR": args.DDS_RESULTS_DIR,
        "HASH_SIZE": args.HASH_SIZE,
        "REWARD_SCALE": args.REWARD_SCALE,
        "ACTOR_ACTIVATION": args.ACTOR_ACTIVATION,
        "ACTOR_MODEL_TYPE": args.ACTOR_MODEL_TYPE,
        "OPP_ACTIVATION": args.OPP_ACTIVATION,
        "OPP_MODEL_TYPE": args.OPP_MODEL_TYPE,
        "OPP_MODEL_PATH": args.OPP_MODEL_PATH,
        "NUM_EVAL_ENVS": args.NUM_EVAL_ENVS,
        "LOAD_INITIAL_MODEL": args.LOAD_INITIAL_MODEL,
        "INITIAL_MODEL_PATH": args.INITIAL_MODEL_PATH,
        "SAVE_MODEL": args.SAVE_MODEL,
        "MODEL_SAVE_PATH": args.MODEL_SAVE_PATH,
        "LOG_PATH": args.LOG_PATH,
        "EXP_NAME": args.EXP_NAME,
        "TRACK": args.TRACK,
        "ACTOR_ILLEGAL_ACTION_MASK": args.ACTOR_ILLEGAL_ACTION_MASK,
        "ACTOR_ILLEGAL_ACTION_PENALTY": args.ACTOR_ILLEGAL_ACTION_PENALTY,
        "ILLEGAL_ACTION_L2NORM_COEF": args.ILLEGAL_ACTION_L2NORM_COEF,
        "GAME_MODE": args.GAME_MODE,
    }
    if args.TRACK:
        key = (
            "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
        )
        wandb.login(key=key)
        wandb.init(
            project=f"ppo-bridge",
            name=args.EXP_NAME,
            config=args.dict(),
            save_code=True,
        )
        os.mkdir(os.path.join(config["LOG_PATH"], config["EXP_NAME"]))
        config_file = open(
            os.path.join(config["LOG_PATH"], config["EXP_NAME"], "config.json"),
            mode="w",
        )
        json.dump(
            config,
            config_file,
            indent=2,
            ensure_ascii=False,
        )
        config_file.close()
    pprint(config)
    print("training of", config["ENV_NAME"])
    rng = jax.random.PRNGKey(0)
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
