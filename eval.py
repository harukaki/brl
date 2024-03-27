import jax
from omegaconf import OmegaConf
from pgx.bridge_bidding import BridgeBidding
import pickle
from src.evaluation import make_simple_duplicate_evaluate
from pydantic import BaseModel


class EVALConfig(BaseModel):
    """
    Configuration settings for evaluating models against each other.

    Attributes:
        team1_model_path    Path to the model for team1.
        team2_model_path    Path to the model for team2.
        team1_activation    Activation function for team1, either 'tanh' or 'relu'.
        team1_model_type    Model type for team1, either 'DeepMind' or 'FAIR'.
        team2_activation    Activation function for team2, either 'tanh' or 'relu'.
        team2_model_type    Model type for team2, either 'DeepMind' or 'FAIR'.
        num_eval_envs       Number of environments for evaluation.
    """
    team1_model_path: str = None
    team2_model_path: str = None
    team1_activation: str = "relu"
    team1_model_type: str = "DeepMind"
    team2_activation: str = "relu"
    team2_model_type: str = "DeepMind"
    num_eval_envs: int = 100


args = EVALConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

if __name__ == "__main__":
    config = {
        "team1_model_path": args.team1_model_path,
        "team2_model_path": args.team2_model_path,
        "team1_activation": args.team1_activation,
        "team1_model_type": args.team1_model_type,
        "team2_activation": args.team2_activation,
        "team2_model_type": args.team2_model_type,
        "num_eval_envs": args.num_eval_envs,
    }
    eval_env = BridgeBidding("dds_results/test_000.npy")
    rng = jax.random.PRNGKey(0)
    duplicate_evaluate = make_simple_duplicate_evaluate(
        eval_env,
        team1_activation=config["team1_activation"],
        team1_model_type=config["team1_model_type"],
        team2_activation=config["team2_activation"],
        team2_model_type=config["team2_model_type"],
        num_eval_envs=config["num_eval_envs"],
    )
    duplicate_evaluate = jax.jit(duplicate_evaluate)

    print(f"num envs: {config['num_eval_envs']}")
    team1_params = pickle.load(open(config["team1_model_path"], "rb"))
    team2_params = pickle.load(open(config["team2_model_path"], "rb"))
    print("---------------------------------------------------")
    print(f'{config["team1_model_path"]} vs. {config["team1_model_path"]}')
    log, tablea_info, tableb_info = duplicate_evaluate(
        team1_params=team1_params,
        team2_params=team2_params,
        rng_key=rng,
    )
    print(f"IMP: {float(log[0])} ± {float(log[1])}")
