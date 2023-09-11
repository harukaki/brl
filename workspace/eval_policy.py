import wandb
from src.models import make_forward_pass
import pgx.bridge_bidding as bb
import jax
import jax.numpy as jnp
from src.evaluation import make_evaluate, make_evaluate_log
from pprint import pprint
import pickle
import os

if __name__ == "__main__":
    config = {
        "ACTOR_ACTIVATION": "relu",
        "ACTOR_MODEL_TYPE": "DeepMind",
        "OPP_ACTIVATION": "relu",
        "OPP_MODEL_TYPE": "DeepMind",
        "OPP_MODEL_PATH": "sl_params/params-300000.pkl",
        "NUM_EVAL_ENVS": 4,
        "LOG_PATH": "",
        "EXP_NAME": "",
        "PARAM_PATH": "sl_params/params-300000.pkl",
        "TRACK": False,
    }
    if config["TRACK"]:
        key = (
            "ffda4b38a6fd57db59331dd7ba8c7e316b179dd9"  # please specify your wandb key
        )
        wandb.login(key=key)
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
    log_info, table_a_info, table_b_info = jax.jit(duplicate_evaluate)(params, rng)
    print(table_a_info)
    print(table_b_info)
    print(log_info)
    # state.save_svg("svg/test_du.svg")
    # state.save_svg("svg/test.svg")
    # print(state)
    log = jax.jit(make_evaluate_log)(log_info)
    pprint(log)

    if config["TRACK"]:
        wandb.log(log)
