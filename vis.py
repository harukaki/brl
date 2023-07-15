import jax
import jax.numpy as jnp
import pgx
import time
import pickle
from ppo_bridge import ActorCritic
import distrax
import argparse
import pgx.bridge_bidding as bb
from ppo_bridge import DeepMind_sl_net_fn
import haiku as hk


sl_forward_pass = hk.without_apply_rng(hk.transform(DeepMind_sl_net_fn))
DeepMind_sl_model_param = pickle.load(
    open("bridge_bidding_sl_networks/params-290000.pkl", "rb")
)


def wraped_sl_forward_pass_apply(observation):
    return sl_forward_pass.apply(DeepMind_sl_model_param, observation)


def net_fn(x):
    """Haiku module for our network."""
    net = hk.Sequential(
        [
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(38),
            jax.nn.log_softmax,
        ]
    )
    return net(x)


net = hk.without_apply_rng(hk.transform(net_fn))

sl_params = pickle.load(
    open("bridge_bidding_sl_networks/params-240000.pkl", "rb")
)


TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def act_sl_model(params, observation):
    policy = jnp.exp(net.apply(params, observation))
    return jnp.argmax(policy, axis=1)


def visualize(network, params, env, rng_key):
    subkeys = jax.random.split(rng_key, 5)
    state = jax.vmap(env.init)(subkeys)
    states = []
    states.append(state)
    step_fn = jax.jit(jax.vmap(env.step))
    while not state.terminated.all():
        logits, value = network.apply(params, state.observation)
        logits = logits + jnp.finfo(jnp.float64).min * (
            ~state.legal_action_mask
        )
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        states.append(state)
    fname = f"vis/{'_'.join((env.id).lower().split())}.svg"
    pgx.save_svg_animation(states, fname, frame_duration_seconds=0.7)


def visualize_vs_sl(network, params, env, rng_key):
    rng_key, _rng = jax.random.split(rng_key)
    subkeys = jax.random.split(_rng, 5)
    state = jax.vmap(env.init)(subkeys)
    states = []
    states.append(state)
    step_fn = jax.jit(jax.vmap(env.step))
    while not state.terminated.all():
        rng_key, _rng = jax.random.split(rng_key)
        logits, value = network.apply(params, state.observation)
        logits = logits + jnp.finfo(jnp.float64).min * (
            ~state.legal_action_mask
        )
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        print(action)
        print(action.shape)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        states.append(state)
        # sl model turn
        # print("===01==")
        # print(f"current player: {state.current_player}")
        action = act_sl_model(sl_params, state.observation)
        # pi = jnp.exp(logits)
        # action = jnp.argmax(pi, axis=1)
        print(state)
        print(action)
        print(action.shape)
        state = step_fn(state, action)
        # step by left
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        # actor teammate turn
        # print("===02==")
        # print(f"current player: {state.current_player}")
        rng_key, _rng = jax.random.split(rng_key)
        logits, value = network.apply(params, state.observation)
        logits = logits + jnp.finfo(jnp.float64).min * (
            ~state.legal_action_mask
        )
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        states.append(state)
        # print(f"actor team, action: {action}")
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===03==")
        # print(f"current player: {state.current_player}")
        action = act_sl_model(sl_params, state.observation)
        # pi = jnp.exp(logits)
        # action = jnp.argmax(pi, axis=1)
        state = step_fn(state, action)
        # step by left
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")
    state.save_svg("vis.svg")
    fname = f"vis/{'_'.join((env.id).lower().split())}.svg"
    pgx.save_svg_animation(states, fname, frame_duration_seconds=0.7)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", type=str, default="bridge_bidding")
    # args = parser.parse_args()

    env = bb.BridgeBidding(dds_results_table_path="100_hash.npy")
    ckpt_filename = "checkpoints/bridge_bidding/model.ckpt"
    with open(ckpt_filename, "rb") as f:
        params = pickle.load(f)["model"][0]
    print(params)
    env = bb.BridgeBidding(dds_results_table_path="100_hash.npy")

    def rl_forward_pass(x, is_eval=False):
        net = ActorCritic(env.num_actions, activation="relu", env_name=env.id)
        logits, value = net(x, is_training=not is_eval, test_local_stats=False)
        return logits, value

    rl_forward_pass = hk.without_apply_rng(hk.transform(rl_forward_pass))
    rng_key = jax.random.PRNGKey(3)
    visualize_vs_sl(rl_forward_pass, params, env, rng_key)
