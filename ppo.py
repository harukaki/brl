"""This code is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple, Literal
from pgx.bridge_bidding import BridgeBidding, download_dds_results
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
    """Configuration settings for PPO (Proximal Policy Optimization) training, evaluation, and logging.

    Attributes
        # rollout settings
        num_envs              Number of parallels for each actor rollout
        num_steps             Number of steps for each actor rollout
        minibatch_size        Minibatch size
        total_timesteps       Number of steps experienced by the end of the training

        # ppo settings
        update_epochs         Number of epochs for ppo update
        lr                    Learning rate for Adam
        gamma                 Discount factor gamma
        gae_lambda            GAE lambda
        clip_eps              Clip for ppo
        ent_coef              Entropy coefficient
        vf_coef               Value loss coefficient

        # evaluation settings
        num_eval_envs         Number of parallels for evaluation
        eval_opp_model_path   Path to the baseline model prepared for evaluation
        num_eval_step         Interval for evaluation

        # other settings
        load_initial_model    Whether to load a pretrained model as the initial values for the neural network
        initial_model_path    Path to the initial model for the neural network
        log_path              Path to the directory where training settings and trained models are saved
        exp_name              Name of experiment
        save_model            Whether to save the trained model
        save_model_interval   Interval for saving the trained model
    """

    seed: int = 0
    lr: float = 0.000001
    num_envs: int = 8192
    num_steps: int = 32
    total_timesteps: int = 2_621_440_000
    update_epochs: int = 10
    minibatch_size: int = 1024
    num_minibatches: int = 128
    num_updates: int = 10000
    # dataset config
    dds_results_dir: str = "dds_results"
    hash_size: int = 100_000
    # eval config
    num_eval_envs: int = 10000
    eval_opp_activation: str = "relu"
    eval_opp_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"
    eval_opp_model_path: str = None
    num_eval_step: int = 10
    # log config
    save_model: bool = True
    save_model_interval: int = 1
    log_path: str = "rl_log"
    exp_name: str = "exp_0000"
    save_model_path: str = "rl_params"
    track: bool = True

    # actor config
    load_initial_model: bool = False
    initial_model_path: str = None
    actor_activation: str = "relu"
    actor_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"
    # opposite config
    game_model: Literal["competitive", "free-run"] = "competitive"
    self_play: bool = True
    opp_activation: str = "relu"
    opp_model_type: Literal["DeepMind", "FAIR"] = "DeepMind"
    opp_model_path: str = None
    ratio_model_zoo: float = 0
    num_model_zoo: int = 50_000
    threshold_model_zoo: float = -24
    prioritized_fictitious: bool = False
    prior_t: float = 1
    num_prioritized_envs: int = 100

    # GAE config
    gamma: float = 1
    gae_lambda: float = 0.95
    # loss config
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    # PPO code optimaization
    value_clipping: bool = True
    global_gradient_clipping: bool = True
    anneal_lr: bool = False  # True
    reward_scaling: bool = False
    max_grad_norm: float = 0.5
    reward_scale: float = 7600

    # illegal action config
    actor_illegal_action_mask: bool = True
    actor_illegal_action_penalty: bool = False
    illegal_action_penalty: float = -1
    illegal_action_l2norm_coef: float = 0


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))


def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["num_minibatches"] * config["update_epochs"]))
        / config["num_updates"]
    )
    return config["lr"] * frac


if args.anneal_lr:
    if args.global_gradient_clipping:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        optimizer = optax.adam(learning_rate=linear_schedule, eps=1e-5)

else:
    if args.global_gradient_clipping:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        )
    else:
        optimizer = optax.adam(args.lr, eps=1e-5)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def train(config, rng):
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["num_minibatches"] = (
        config["num_envs"] * config["num_steps"] // config["minibatch_size"]
    )
    if not os.path.isdir("dds_results"):
        download_dds_results()
    env = BridgeBidding()

    actor_forward_pass = make_forward_pass(
        activation=config["actor_activation"],
        model_type=config["actor_model_type"],
    )
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = actor_forward_pass.init(_rng, init_x)  # params  # DONE
    opt_state = optimizer.init(params=params)  # DONE

    # LOAD INITIAL MODEL
    if config["load_initial_model"]:
        params = pickle.load(open(config["initial_model_path"], "rb"))
        print(f"load initial params for actor: {config['initial_model_path']}")

    # MAKE EVAL
    rng, eval_rng = jax.random.split(rng)
    eval_env = BridgeBidding("dds_results/test_000.npy")
    simple_evaluate = make_simple_evaluate(
        eval_env=eval_env,
        team1_activation=config["actor_activation"],
        team1_model_type=config["actor_model_type"],
        team2_activation=config["eval_opp_activation"],
        team2_model_type=config["eval_opp_model_type"],
        team2_model_path=config["eval_opp_model_path"],
        num_eval_envs=config["num_eval_envs"],
    )
    simple_duplicate_evaluate = make_simple_duplicate_evaluate(
        eval_env=eval_env,
        team1_activation=config["actor_activation"],
        team1_model_type=config["actor_model_type"],
        team2_activation=config["actor_activation"],
        team2_model_type=config["actor_model_type"],
        num_eval_envs=config["num_prioritized_envs"],
    )
    duplicate_evaluate = make_evaluate(
        eval_env=eval_env,
        team1_activation=config["actor_activation"],
        team1_model_type=config["actor_model_type"],
        team2_activation=config["eval_opp_activation"],
        team2_model_type=config["eval_opp_model_type"],
        team2_model_path=config["eval_opp_model_path"],
        num_eval_envs=config["num_eval_envs"],
        game_mode=config["game_model"],
        duplicate=True,
    )
    jit_simple_evaluate = jax.jit(simple_evaluate)
    jit_simple_duplicate_evaluate = jax.jit(simple_duplicate_evaluate)
    jit_diplicate_evaluate = jax.jit(duplicate_evaluate)
    jit_make_evaluate_log = jax.jit(make_evaluate_log)

    # INIT UPDATE FUNCTION

    opp_forward_pass = make_forward_pass(
        activation=config["opp_activation"],
        model_type=config["opp_model_type"],
    )

    # INIT ENV
    env_list = []
    init_list = []
    roll_out_list = []
    train_dds_results_list = sorted(
        [path for path in os.listdir(config["dds_results_dir"]) if "train" in path]
    )

    # dds_resultsの異なるhash tableをloadしたenvを用意
    for file in train_dds_results_list:
        env = BridgeBidding(os.path.join(config["dds_results_dir"], file))
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
    reset_rng = jax.random.split(_rng, config["num_envs"])
    init = init_list[0]
    roll_out = roll_out_list[0]
    env_state = init(reset_rng)

    hash_index_list = np.arange(len(train_dds_results_list))
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

    if not config["self_play"]:
        opp_params = pickle.load(open(config["eval_opp_model_path"], "rb"))
    else:
        opp_params = params
    if config["save_model"]:
        os.mkdir(
            os.path.join(
                config["log_path"],
                config["exp_name"],
                config["save_model_path"],
            )
        )
    print("start training")
    for i in range(config["num_updates"]):
        print(f"--------------iteration {i}---------------")
        # save model
        if (i != 0) and (i % config["save_model_interval"] == 0):
            if config["save_model"]:
                with open(
                    os.path.join(
                        config["log_path"],
                        config["exp_name"],
                        config["save_model_path"],
                        f"params-{i:08}.pkl",
                    ),
                    "wb",
                ) as writer:
                    pickle.dump(runner_state[0], writer)

        # eval
        time_eval_sta = time.time()
        R = jit_simple_evaluate(runner_state[0], eval_rng)
        time_eval_end = time.time()
        print(f"eval time: {time_eval_end-time_eval_sta}")
        if i % config["num_eval_step"] == 0:
            time_du_sta = time.time()
            log_info, _, _ = jit_diplicate_evaluate(runner_state[0], eval_rng)
            eval_log = jit_make_evaluate_log(log_info)
            time_du_end = time.time()
            print(f"duplicate eval time: {time_du_end-time_du_sta}")

        if config["self_play"]:
            (imp_opp, _, _), _, _ = jit_simple_duplicate_evaluate(
                team1_params=runner_state[0],
                team2_params=opp_params,
                rng_key=eval_rng,
            )
            if imp_opp >= config["threshold_model_zoo"]:
                params_list = sorted(
                    [
                        path
                        for path in os.listdir(
                            os.path.join(
                                config["log_path"],
                                config["exp_name"],
                                config["save_model_path"],
                            )
                        )
                        if "params" in path
                    ]
                )
                if (len(params_list) != 0) and np.random.binomial(
                    size=1, n=1, p=config["ratio_model_zoo"]
                ):
                    if config["prioritized_fictitious"]:
                        league_sta = time.time()
                        win_rate_list = np.zeros(len(params_list))
                        imp_list = np.zeros(len(params_list))
                        team1_params = runner_state[0]
                        for i in range(len(params_list)):
                            team2_params = pickle.load(
                                open(
                                    os.path.join(
                                        config["log_path"],
                                        config["exp_name"],
                                        config["save_model_path"],
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
                                / config["prior_t"]
                            )
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
                                config["log_path"],
                                config["exp_name"],
                                config["save_model_path"],
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
        steps += config["num_envs"] * config["num_steps"]

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
                    (i + 1) * config["update_epochs"] * config["num_minibatches"]
                )
            ),
            "train/imp_opp_before": float(imp_opp_before),
            "train/imp_opp_after": float(imp_opp_after),
            "board_num": int(runner_state[4]),
            "steps": steps,
        }
        pprint(log)
        if i % config["num_eval_step"] == 0:
            log = {**log, **eval_log}
        if config["track"]:
            wandb.log(log)
        if (runner_state[4] - board_count) // config["hash_size"] >= 1:
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
            reset_rng = jax.random.split(_rng, config["num_envs"])

            env_state = init(reset_rng)
            runner_state = (
                runner_state[0],
                runner_state[1],
                env_state,
                env_state.observation,
                runner_state[4],
                _rng,
            )
    if config["save_model"]:
        with open(
            os.path.join(
                config["log_path"],
                config["exp_name"],
                config["save_model_path"],
                f"params-{i + 1:08}.pkl",
            ),
            "wb",
        ) as writer:
            pickle.dump(runner_state[0], writer)
        with open(
            os.path.join(
                config["log_path"],
                config["exp_name"],
                config["save_model_path"],
                f"opt_state-{i + 1:08}.pkl",
            ),
            "wb",
        ) as writer:
            pickle.dump(runner_state[1], writer)

    return runner_state


if __name__ == "__main__":
    config = {
        "lr": args.lr,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "total_timesteps": args.total_timesteps,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "anneal_lr": True,
        "num_eval_envs": args.num_eval_envs,
        "dds_results_dir": args.dds_results_dir,
        "hash_size": args.hash_size,
        "reward_scale": args.reward_scale,
        "actor_activation": args.actor_activation,
        "actor_model_type": args.actor_model_type,
        "opp_activation": args.opp_activation,
        "opp_model_type": args.opp_model_type,
        "opp_model_path": args.opp_model_path,
        "load_initial_model": args.load_initial_model,
        "initial_model_path": args.initial_model_path,
        "save_model": args.save_model,
        "save_model_interval": args.save_model_interval,
        "save_model_path": args.save_model_path,
        "log_path": args.log_path,
        "exp_name": args.exp_name,
        "track": args.track,
        "actor_illegal_action_mask": args.actor_illegal_action_mask,
        "actor_illegal_action_penalty": args.actor_illegal_action_penalty,
        "illegal_action_l2norm_coef": args.illegal_action_l2norm_coef,
        "game_model": args.game_model,
        "reward_scaling": args.reward_scaling,
        "seed": args.seed,
        "self_play": args.self_play,
        "ratio_model_zoo": args.ratio_model_zoo,
        "threshold_model_zoo": args.threshold_model_zoo,
        "prioritized_fictitious": args.prioritized_fictitious,
        "prior_t": args.prior_t,
        "num_prioritized_envs": args.num_prioritized_envs,
        "num_model_zoo": args.num_model_zoo,
        "eval_opp_activation": args.eval_opp_activation,
        "eval_opp_model_type": args.eval_opp_model_type,
        "eval_opp_model_path": args.eval_opp_model_path,
        "num_eval_step": args.num_eval_step,
        "value_clipping": args.value_clipping,
        "global_gradient_clipping": args.global_gradient_clipping,
    }
    if args.track:
        wandb.init(
            project="ppo-bridge",
            name=args.exp_name,
            config=args.model_dump(),
            save_code=True,
        )
        os.mkdir(os.path.join(config["log_path"], config["exp_name"]))
        config_file = open(
            os.path.join(config["log_path"], config["exp_name"], "config.json"),
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
    rng = jax.random.PRNGKey(config["seed"])
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
