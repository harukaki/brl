"""This code is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple, Literal
import pgx.bridge_bidding as bb
import time
import os
import json
from pprint import pprint


import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb

from src.models import make_forward_pass
from src.evaluation import (
    make_evaluate,
    make_evaluate_log,
    make_simple_evaluate,
    make_simple_duplicate_evaluate,
)
from src.roll_out import make_roll_out
from src.gae import make_calc_gae
from src.update import make_update_step


print(jax.default_backend())
print(jax.local_devices())


class PPOConfig(BaseModel):
    SEED: int = 0
    LR: float = 0.000001  # 0.0003
    NUM_ENVS: int = 8192
    NUM_STEPS: int = 32
    TOTAL_TIMESTEPS: int = 2_621_440_000
    UPDATE_EPOCHS: int = 10  # 一回のupdateでbatchが何回学習されるか
    NUM_MINIBATCHES: int = 128
    NUM_UPDATES: int = 10000  # updateが何回されるか　TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENV
    MINIBATCH_SIZE: int = 1024  # update中の1 epochで使用される数
    # dataset config
    DDS_RESULTS_DIR: str = "dds_results"
    HASH_SIZE: int = 100_000
    TRAIN_SIZE: int = 12_500_000
    HASH_TABLE_NUM: int = 125
    # eval config
    NUM_EVAL_ENVS: int = 10000
    EVAL_OPP_ACTIVATION: str = "relu"
    EVAL_OPP_MODEL_TYPE: Literal[
        "DeepMind", "FAIR", "FAIR_6", "DeepMind_6", "DeepMind_8"
    ] = "DeepMind"
    EVAL_OPP_MODEL_PATH: str = "sl_log/sl_deepmind/params-400000.pkl"
    NUM_EVAL_STEP: int = 10
    # log config
    SAVE_MODEL: bool = False
    SAVE_MODEL_INTERVAL: int = 1
    LOG_PATH: str = "rl_log"
    EXP_NAME: str = "exp_0000"
    MODEL_SAVE_PATH: str = "rl_params"
    TRACK: bool = False

    # actor config
    LOAD_INITIAL_MODEL: bool = False
    INITIAL_MODEL_PATH: str = "sl_log/sl_deepmind_actor_critic/params-400000.pkl"
    ACTOR_ACTIVATION: str = "relu"
    ACTOR_MODEL_TYPE: Literal[
        "DeepMind", "FAIR", "FAIR_6", "DeepMind_6", "DeepMind_8"
    ] = "DeepMind"
    # opposite config
    GAME_MODE: Literal["competitive", "free-run"] = "competitive"
    SELF_PLAY: bool = True
    OPP_ACTIVATION: str = "relu"
    OPP_MODEL_TYPE: Literal[
        "DeepMind", "FAIR", "FAIR_6", "DeepMind_6", "DeepMind_8"
    ] = "DeepMind"
    OPP_MODEL_PATH: str = None
    MODEL_ZOO_RATIO: float = 0
    MODEL_ZOO_NUM: int = 50_000
    MODEL_ZOO_THRESHOLD: float = -24
    PRIORITIZED_FICTITIOUS: bool = False
    PRIOR_T: float = 1
    NUM_PRIORITIZED_ENVS: int = 100

    # GAE config
    GAMMA: float = 1
    GAE_LAMBDA: float = 0.95
    # loss config
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.001
    VF_COEF: float = 0.5
    # PPO code optimaization
    VALUE_CLIPPING: bool = True
    GLOBAL_GRADIENT_CLIPPING: bool = True
    ANNEAL_LR: bool = False  # True
    REWARD_SCALING: bool = False
    MAX_GRAD_NORM: float = 0.5
    REWARD_SCALE: float = 7600

    # illegal action config
    ACTOR_ILLEGAL_ACTION_MASK: bool = True
    ACTOR_ILLEGAL_ACTION_PENALTY: bool = False
    ILLEGAL_ACTION_PENALTY: float = -1
    ILLEGAL_ACTION_L2NORM_COEF: float = 0


def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
if args.ANNEAL_LR:
    if args.GLOBAL_GRADIENT_CLIPPING:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.MAX_GRAD_NORM),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        optimizer = optax.adam(learning_rate=linear_schedule, eps=1e-5)

else:
    if args.GLOBAL_GRADIENT_CLIPPING:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.MAX_GRAD_NORM),
            optax.adam(args.LR, eps=1e-5),
        )
    else:
        optimizer = optax.adam(args.LR, eps=1e-5)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def _get(x, i):
    return x[i]


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
    simple_evaluate = make_simple_evaluate(config)
    simple_duplicate_evaluate = make_simple_duplicate_evaluate(config)
    duplicate_evaluate = make_evaluate(config, duplicate=True)
    jit_simple_evaluate = jax.jit(simple_evaluate)
    jit_simple_duplicate_evaluate = jax.jit(simple_duplicate_evaluate)
    jit_diplicate_evaluate = jax.jit(duplicate_evaluate)
    jit_make_evaluate_log = jax.jit(make_evaluate_log)

    # INIT UPDATE FUNCTION

    opp_forward_pass = make_forward_pass(
        activation=config["OPP_ACTIVATION"],
        model_type=config["OPP_MODEL_TYPE"],
    )

    # INIT ENV
    env_list = []
    init_list = []
    roll_out_list = []
    train_dds_results_list = sorted(
        [path for path in os.listdir(config["DDS_RESULTS_DIR"]) if "train" in path]
    )[: config["HASH_TABLE_NUM"]]

    # dds_resultsの異なるhash tableをloadしたenvを用意
    for file in train_dds_results_list:
        env = bb.BridgeBidding(os.path.join(config["DDS_RESULTS_DIR"], file))
        env_list.append(env)
        init_list.append(jax.jit(jax.vmap(env.init)))
        roll_out_list.append(
            jax.jit(make_roll_out(config, env, actor_forward_pass, opp_forward_pass))
        )
    calc_gae = jax.jit(make_calc_gae(config, actor_forward_pass))
    update_step = jax.jit(
        make_update_step(config, actor_forward_pass, optimizer=optimizer)
    )

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    init = init_list[0]
    roll_out = roll_out_list[0]
    env_state = init(reset_rng)

    hash_index_list = np.arange(config["HASH_TABLE_NUM"])
    steps = 0
    hash_index = 0
    board_count = 0
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

    if not config["SELF_PLAY"]:
        opp_params = pickle.load(open(config["EVAL_OPP_MODEL_PATH"], "rb"))
    else:
        opp_params = params
    if config["SAVE_MODEL"]:
        os.mkdir(
            os.path.join(
                config["LOG_PATH"],
                config["EXP_NAME"],
                config["MODEL_SAVE_PATH"],
            )
        )

    for i in range(config["NUM_UPDATES"]):
        # save model
        if (i != 0) and (i % config["SAVE_MODEL_INTERVAL"] == 0):
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

        # eval
        time_eval_sta = time.time()
        R = jit_simple_evaluate(runner_state[0], eval_rng)
        time_eval_end = time.time()
        print(f"eval time: {time_eval_end-time_eval_sta}")
        if i % config["NUM_EVAL_STEP"] == 0:
            time_du_sta = time.time()
            log_info, _, _ = jit_diplicate_evaluate(runner_state[0], eval_rng)
            eval_log = jit_make_evaluate_log(log_info)
            time_du_end = time.time()
            print(f"duplicate eval time: {time_du_end-time_du_sta}")

        if config["SELF_PLAY"]:
            (imp_opp, _, _), _, _ = jit_simple_duplicate_evaluate(
                team1_params=runner_state[0],
                team2_params=opp_params,
                rng_key=eval_rng,
            )
            if imp_opp >= config["MODEL_ZOO_THRESHOLD"]:
                params_list = sorted(
                    [
                        path
                        for path in os.listdir(
                            os.path.join(
                                config["LOG_PATH"],
                                config["EXP_NAME"],
                                config["MODEL_SAVE_PATH"],
                            )
                        )
                        if "params" in path
                    ]
                )
                if (len(params_list) != 0) and np.random.binomial(
                    size=1, n=1, p=config["MODEL_ZOO_RATIO"]
                ):
                    if config["PRIORITIZED_FICTITIOUS"]:
                        league_sta = time.time()
                        win_rate_list = np.zeros(len(params_list))
                        imp_list = np.zeros(len(params_list))
                        team1_params = runner_state[0]
                        for i in range(len(params_list)):
                            team2_params = pickle.load(
                                open(
                                    os.path.join(
                                        config["LOG_PATH"],
                                        config["EXP_NAME"],
                                        config["MODEL_SAVE_PATH"],
                                        params_list[i],
                                    ),
                                    "rb",
                                )
                            )
                            log, _, _ = jit_simple_duplicate_evaluate(
                                team1_params=team1_params,
                                team2_params=team2_params,
                                rng_key=eval_rng,
                            )
                            win_rate_list[i] = log[2]
                            imp_list[i] = log[0]
                        league_end = time.time()
                        print(f"league time: {league_end - league_sta}")

                        def softmax(x):
                            exp_values = np.exp(
                                (x - np.max(x, axis=-1, keepdims=True))
                                / config["PRIOR_T"]
                            )  # 数値安定性のために最大値を引く
                            probabilities = exp_values / np.sum(
                                exp_values, axis=-1, keepdims=True
                            )
                            return probabilities

                        probabilities = softmax(-imp_list)
                        params_index = np.random.choice(
                            len(probabilities), p=probabilities
                        )
                        params_path = params_list[params_index]
                    else:
                        params_path = np.random.choice(params_list)
                    print(f"opposite params: {params_path}")
                    opp_params = pickle.load(
                        open(
                            os.path.join(
                                config["LOG_PATH"],
                                config["EXP_NAME"],
                                config["MODEL_SAVE_PATH"],
                                params_path,
                            ),
                            "rb",
                        )
                    )
                else:
                    print("opposite params: latest")
                    opp_params = runner_state[0]
            else:
                print("opposite params: latest")
                opp_params = opp_params
        (imp_opp_before, _, _), _, _ = jit_simple_duplicate_evaluate(
            team1_params=runner_state[0],
            team2_params=opp_params,
            rng_key=eval_rng,
        )
        time1 = time.time()
        runner_state, traj_batch = roll_out(
            runner_state=runner_state, opp_params=opp_params
        )
        time2 = time.time()
        advantages, targets = calc_gae(runner_state=runner_state, traj_batch=traj_batch)
        time3 = time.time()
        runner_state, loss_info = update_step(
            runner_state=runner_state,
            traj_batch=traj_batch,
            advantages=advantages,
            targets=targets,
        )
        time4 = time.time()
        (imp_opp_after, _, _), _, _ = jit_simple_duplicate_evaluate(
            team1_params=runner_state[0],
            team2_params=opp_params,
            rng_key=eval_rng,
        )

        print(f"rollout time: {time2 - time1}")
        print(f"calc gae time: {time3 - time2}")
        print(f"update time: {time4 - time3}")
        steps += config["NUM_ENVS"] * config["NUM_STEPS"]

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            approx_kl,
            clipflacs,
            illegal_action_loss,
        ) = loss_info

        # make log
        log = {
            "train/score": float(R),
            "train/total_loss": float(total_loss[-1][-1]),
            "train/value_loss": float(value_loss[-1][-1]),
            "train/loss_actor": float(loss_actor[-1][-1]),
            "train/illegal_action_loss": float(illegal_action_loss[-1][-1]),
            "train/policy_entropy": float(entropy[-1][-1]),
            "train/clipflacs": float(clipflacs[-1][-1]),
            "train/approx_kl": float(approx_kl[-1][-1]),
            "train/lr": float(
                linear_schedule(
                    (i + 1) * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
                )
            ),
            "train/imp_opp_before": float(imp_opp_before),
            "train/imp_opp_after": float(imp_opp_after),
            "board_num": int(runner_state[4]),
            "steps": steps,
        }
        pprint(log)
        if i % config["NUM_EVAL_STEP"] == 0:
            log = {**log, **eval_log}
        if config["TRACK"]:
            wandb.log(log)
        if (runner_state[4] - board_count) // config["HASH_SIZE"] >= 1:
            hash_index += 1
            print(f"board count: {runner_state[4] - board_count}")
            board_count = runner_state[4]
            if hash_index == len(hash_index_list):
                hash_index = 0
                print("use all hash, shuffle")
                np.random.shuffle(hash_index_list)
            print(
                f"use hash table: {train_dds_results_list[hash_index_list[hash_index]]}"
            )
            init = init_list[hash_index_list[hash_index]]
            roll_out = roll_out_list[hash_index_list[hash_index]]
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

            env_state = init(reset_rng)
            runner_state = (
                runner_state[0],
                runner_state[1],
                env_state,
                env_state.observation,
                runner_state[4],
                _rng,
            )
    if config["SAVE_MODEL"]:
        with open(
            os.path.join(
                config["LOG_PATH"],
                config["EXP_NAME"],
                config["MODEL_SAVE_PATH"],
                f"params-{i + 1:08}.pkl",
            ),
            "wb",
        ) as writer:
            pickle.dump(runner_state[0], writer)
        with open(
            os.path.join(
                config["LOG_PATH"],
                config["EXP_NAME"],
                config["MODEL_SAVE_PATH"],
                f"opt_state-{i + 1:08}.pkl",
            ),
            "wb",
        ) as writer:
            pickle.dump(runner_state[1], writer)

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
        "ANNEAL_LR": True,
        "NUM_EVAL_ENVS": args.NUM_EVAL_ENVS,
        "DDS_RESULTS_DIR": args.DDS_RESULTS_DIR,
        "HASH_SIZE": args.HASH_SIZE,
        "REWARD_SCALE": args.REWARD_SCALE,
        "ACTOR_ACTIVATION": args.ACTOR_ACTIVATION,
        "ACTOR_MODEL_TYPE": args.ACTOR_MODEL_TYPE,
        "OPP_ACTIVATION": args.OPP_ACTIVATION,
        "OPP_MODEL_TYPE": args.OPP_MODEL_TYPE,
        "OPP_MODEL_PATH": args.OPP_MODEL_PATH,
        "LOAD_INITIAL_MODEL": args.LOAD_INITIAL_MODEL,
        "INITIAL_MODEL_PATH": args.INITIAL_MODEL_PATH,
        "SAVE_MODEL": args.SAVE_MODEL,
        "SAVE_MODEL_INTERVAL": args.SAVE_MODEL_INTERVAL,
        "MODEL_SAVE_PATH": args.MODEL_SAVE_PATH,
        "LOG_PATH": args.LOG_PATH,
        "EXP_NAME": args.EXP_NAME,
        "TRACK": args.TRACK,
        "ACTOR_ILLEGAL_ACTION_MASK": args.ACTOR_ILLEGAL_ACTION_MASK,
        "ACTOR_ILLEGAL_ACTION_PENALTY": args.ACTOR_ILLEGAL_ACTION_PENALTY,
        "ILLEGAL_ACTION_L2NORM_COEF": args.ILLEGAL_ACTION_L2NORM_COEF,
        "GAME_MODE": args.GAME_MODE,
        "REWARD_SCALING": args.REWARD_SCALING,
        "SEED": args.SEED,
        "SELF_PLAY": args.SELF_PLAY,
        "MODEL_ZOO_RATIO": args.MODEL_ZOO_RATIO,
        "MODEL_ZOO_THRESHOLD": args.MODEL_ZOO_THRESHOLD,
        "PRIORITIZED_FICTITIOUS": args.PRIORITIZED_FICTITIOUS,
        "PRIOR_T": args.PRIOR_T,
        "NUM_PRIORITIZED_ENVS": args.NUM_PRIORITIZED_ENVS,
        "MODEL_ZOO_NUM": args.MODEL_ZOO_NUM,
        "EVAL_OPP_ACTIVATION": args.EVAL_OPP_ACTIVATION,
        "EVAL_OPP_MODEL_TYPE": args.EVAL_OPP_MODEL_TYPE,
        "EVAL_OPP_MODEL_PATH": args.EVAL_OPP_MODEL_PATH,
        "HASH_TABLE_NUM": args.HASH_TABLE_NUM,
        "NUM_EVAL_STEP": args.NUM_EVAL_STEP,
        "VALUE_CLIPPING": args.VALUE_CLIPPING,
        "GLOBAL_GRADIENT_CLIPPING": args.GLOBAL_GRADIENT_CLIPPING,
    }
    if args.TRACK:
        wandb.init(
            project="ppo-bridge",
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
    rng = jax.random.PRNGKey(config["SEED"])
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
