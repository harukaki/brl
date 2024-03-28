import pickle
from src.evaluation import make_simple_duplicate_evaluate
from pgx.bridge_bidding import BridgeBidding
import jax
import numpy as np
import time
import os
import itertools
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel
from omegaconf import OmegaConf


class SelfplayLeagueConfig(BaseModel):
    """各学習における自身の過去モデル間のリーグ対戦による評価の設定

    Attributes
        models_directory            各学習のディレクトリをまとめたディレクトリのパス
        exp_name                    各学習のモデルを保存するディレクトリの名前
        num_eval_envs               デュープリケート対戦のボード数
        skip_interval               リーグ対戦に用いる過去モデルが保存されたstepも間隔
        max_step                    リーグ対戦に用いるモデルが保存された最大step数
        save_fig_directory_path     リーグ対戦の結果の図を保存するディレクトリのパス
        activation                  モデルの活性化関数
        model_type                  モデルのタイプ
    """

    models_directory: str = "models"
    exp_name: str = "pretrained-rl-with-sp"
    num_eval_envs: int = 100
    skip_interval: int = 100
    max_step: int = 10000
    save_fig_directory_path: str = ""
    activation: str = "relu"
    model_type: str = "DeepMind"


args = SelfplayLeagueConfig(**OmegaConf.to_object(OmegaConf.from_cli()))


if __name__ == "__main__":
    eval_env = BridgeBidding("dds_results/test_000.npy")
    rng = jax.random.PRNGKey(0)
    duplicate_evaluate = make_simple_duplicate_evaluate(
        eval_env,
        team1_activation=args.activation,
        team1_model_type=args.model_type,
        team2_activation=args.activation,
        team2_model_type=args.model_type,
        num_eval_envs=args.num_eval_envs,
    )
    duplicate_evaluate = jax.jit(duplicate_evaluate)
    params_list = sorted(
        [
            path
            for path in os.listdir(
                os.path.join(
                    args.models_directory,
                    args.exp_name,
                )
            )
            if "params" in path
            and int(path.split("-")[1].split(".")[0]) % args.skip_interval == 0
            and int(path.split("-")[1].split(".")[0]) <= args.max_step
        ]
    )
    print(params_list)
    match_list = itertools.combinations(range(len(params_list)), 2)
    warnings.simplefilter("ignore")

    win_lose = np.zeros((len(params_list), len(params_list)))
    win_lose_dis = np.zeros((len(params_list), len(params_list)))
    win_lose_clip = np.zeros((len(params_list), len(params_list)))
    print("league match start")
    for match in match_list:
        team1_params = pickle.load(
            open(
                os.path.join(
                    args.models_directory,
                    args.exp_name,
                    params_list[match[0]],
                ),
                "rb",
            )
        )
        team2_params = pickle.load(
            open(
                os.path.join(
                    args.models_directory,
                    args.exp_name,
                    params_list[match[1]],
                ),
                "rb",
            )
        )
        print(f"{match[0]} vs. {match[1]}")
        time1 = time.time()
        log, _, _ = duplicate_evaluate(
            team1_params=team1_params,
            team2_params=team2_params,
            rng_key=rng,
        )
        time2 = time.time()
        imp, se, win_rate = log

        print(f"imp: {imp}")
        print(f"win rate: {win_rate}")
        print(f"time: {time2-time1}")
        win_lose[match[0]][match[1]] = -imp
        win_lose[match[1]][match[0]] = imp
        clip_imp = np.clip(imp, -1, 1)
        win_lose_clip[match[0]][match[1]] = -clip_imp
        win_lose_clip[match[1]][match[0]] = clip_imp
        if imp > 0:
            win_lose_dis[match[0]][match[1]] = -1
            win_lose_dis[match[1]][match[0]] = 1
        elif imp < 0:
            win_lose_dis[match[0]][match[1]] = 1
            win_lose_dis[match[1]][match[0]] = -1
    print("---------------------------------------------------")
    plt.figure()
    sns.heatmap(win_lose, cmap="bwr_r")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.savefig(
        os.path.join(args.save_fig_directory_path, f"{args.exp_name}_league.png")
    )

    plt.figure()
    sns.heatmap(win_lose_dis, cmap="bwr_r")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.savefig(
        os.path.join(args.save_fig_directory_path, f"{args.exp_name}_league_dis.png")
    )

    plt.figure()
    sns.heatmap(win_lose_clip, cmap="bwr_r")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.xlabel(r"Step ($\times {10}^2$)")
    plt.savefig(
        os.path.join(args.save_fig_directory_path, f"{args.exp_name}_league_clip.png")
    )
