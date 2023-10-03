import jax
import jax.numpy as jnp
import os
import pgx.bridge_bidding as bb
from src.evaluation import make_evaluate, make_evaluate_log

import pickle
import wandb
from src.models import make_forward_pass
from pprint import pprint

if __name__ == "__main__":
    key = "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
    wandb.login(key=key)
    config = {
        "ACTOR_ACTIVATION": "relu",
        "ACTOR_MODEL_TYPE": "FAIR",
        "OPP_ACTIVATION": "relu",
        "OPP_MODEL_TYPE": "DeepMind",
        "OPP_MODEL_PATH": "sl_log/sl_deepmind/params-400000.pkl",
        "NUM_EVAL_ENVS": 100,
        "LOG_PATH": "rl_log",
        "EXP_NAME": "exp0090",
        "PARAM_PATH": "rl_params/params-00000289.pkl",
        "TRACK": True,
        "GAME_MODE": "competitive",
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

    if config["TRACK"]:
        wandb.log(log)
