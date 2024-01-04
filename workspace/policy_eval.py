import jax
import jax.numpy as jnp
import os
import pgx.bridge_bidding as bb
from src.evaluation import make_evaluate, make_evaluate_log

import pickle
import wandb
from src.models import make_forward_pass
from pprint import pprint
from src.visualizer import Visualizer
from pgx.bridge_bidding import BridgeBidding, _calculate_dds_tricks

if __name__ == "__main__":
    key = "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
    wandb.login(key=key)
    config = {
        "ACTOR_ACTIVATION": "relu",
        "ACTOR_MODEL_TYPE": "FAIR",
        "EVAL_OPP_ACTIVATION": "relu",
        "EVAL_OPP_MODEL_TYPE": "FAIR",
        # "EVAL_OPP_MODEL_PATH": "sl_log/sl_deepmind/params-400000.pkl",
        "EVAL_OPP_MODEL_PATH": "rl_log/exp0091/rl_params/params-00000761.pkl",
        "NUM_EVAL_ENVS": 4,
        "LOG_PATH": "rl_log",
        "EXP_NAME": "exp0091",
        "PARAM_PATH": "rl_params/params-00000761.pkl",
        "TRACK": False,
        "GAME_MODE": "competitive",
        "SVG_NAME": "1.svg",
    }
    if config["TRACK"]:
        wandb.init(project="eval_test", config=config)
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )
    env = bb.BridgeBidding()
    rng = jax.random.PRNGKey(99)
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
    duplicate_evaluate = make_evaluate(config, duplicate=True)
    print("start")
    log_info, table_a_info, table_b_info = jax.jit(duplicate_evaluate)(params, _rng)
    print(make_evaluate_log(log_info))
    print(table_a_info)
    print(table_b_info)
    log = jax.jit(make_evaluate_log)(log_info)
    pprint(log)
    evaluation = make_evaluate(config)
    state, log_info = jax.jit(evaluation)(params, _rng)
    pprint(log_info)
    v = Visualizer()
    eval_env = bb.BridgeBidding("dds_results/test_000.npy")
    v.get_dwg(states=state, env=eval_env).saveas(
        os.path.join(config["LOG_PATH"], config["EXP_NAME"], config["SVG_NAME"])
    )
    print(state)
    if config["TRACK"]:
        wandb.log(log)
