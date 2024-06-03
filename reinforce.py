import sys
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple, Literal
import distrax
import pgx
from pgx.experimental import auto_reset
import time

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class A2CConfig(BaseModel):
    env_name: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-breakout"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 64
    num_eval_envs: int = 100
    num_steps: int = 64
    total_timesteps: int = 20000000
    update_epochs: int = 1
    batch_size: int = 4096
    # GAE config
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # entropy regularization
    ent_coef: float = 0.01
    # value loss
    vf_coef: float = 0.5
    # PPO code optimaizatino
    reward_scaling: bool = False
    global_gradient_clipping: bool = False
    max_grad_norm: float = 0.5
    # log config
    wandb_project: str = "pgx-minatar-ppo"
    save_model: bool = False

    class Config:
        extra = "forbid"


args = A2CConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)
env = pgx.make(str(args.env_name))

num_updates = args.total_timesteps // args.num_envs // args.num_steps


class ActorCritic(hk.Module):
    def __init__(self, num_actions, activation="tanh"):
        super().__init__()
        self.num_actions = num_actions
        self.activation = activation
        assert activation in ["relu", "tanh"]

    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        x = hk.Conv2D(32, kernel_shape=2)(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        actor_mean = hk.Linear(64)(x)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(64)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(self.num_actions)(actor_mean)

        critic = hk.Linear(64)(x)
        critic = activation(critic)
        critic = hk.Linear(64)(critic)
        critic = activation(critic)
        critic = hk.Linear(1)(critic)

        return actor_mean, jnp.squeeze(critic, axis=-1)


def forward_fn(x, is_eval=False):
    net = ActorCritic(env.num_actions, activation="tanh")
    logits, value = net(x)
    return logits, value


forward = hk.without_apply_rng(hk.transform(forward_fn))


# code optimaization
if args.global_gradient_clipping:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5)
    )
else:
    optimizer = optax.adam(args.lr, eps=1e-5)
if args.reward_scaling:

    def reward_scale(gae):
        return (gae - gae.mean()) / (gae.std() + 1e-8)

    reward_scaling = reward_scale
else:

    def no_scale(gae):
        return gae

    reward_scaling = no_scale


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


"""
def roll_out(runner_state):
    # COLLECT TRAJECTORIES
    step_fn = jax.vmap(auto_reset(env.step, env.init))

    transitions = []

    def cond_fn(runner_state):
        (_, _, env_state, _, _, _, _) = runner_state
        return ~env_state.terminated.all()

    def _env_step(runner_state, unused):
        params, opt_state, env_state, last_obs, rng = runner_state
        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        logits, value = forward.apply(params, last_obs)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        env_state = step_fn(env_state, action)
        transition = Transition(
            env_state.terminated,
            action,
            value,
            jnp.squeeze(env_state.rewards),
            log_prob,
            last_obs,
        )
        runner_state = (params, opt_state, env_state, env_state.observation, rng)
        return runner_state, transition

    # runner_state = jax.lax.while_loop(conf_fn, _env_step, runner_state)
    while cond_fn(runner_state):
        runner_state, transition = _env_step(runner_state, None)
        transitions.append(transition)
    return runner_state
"""


def roll_out(runner_state):
    # COLLECT TRAJECTORIES
    step_fn = jax.vmap(env.step)
    transitions = []

    def cond_fn(tup):
        (
            _,
            _,
            env_state,
            _,
            _,
        ) = tup
        return ~env_state.terminated.all()

    def _env_step(tup):
        runner_state = tup
        params, opt_state, env_state, last_obs, rng = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        logits, value = forward.apply(params, last_obs)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        env_state = step_fn(env_state, action)
        transition = Transition(
            env_state.terminated,
            action,
            value,
            jnp.squeeze(env_state.rewards),
            log_prob,
            last_obs,
        )
        runner_state = (params, opt_state, env_state, env_state.observation, rng)
        transitions.append(transition)
        return runner_state

    runner_state = jax.lax.while_loop(cond_fn, _env_step, runner_state)
    return runner_state, transitions


def calc_gae(runner_state, traj_batch):
    params, opt_state, env_state, last_obs, rng = runner_state
    _, last_val = forward.apply(params, last_obs)

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + args.gamma * next_value * (1 - done) - value
            gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    advantages, targets = _calculate_gae(traj_batch, last_val)
    return advantages, targets


def uptate_step(runner_state, traj_batch, advantages, targets):
    params, opt_state, env_state, last_obs, rng = runner_state

    # UPDATE NETWORK
    def _update_batch(tup, batch_info):
        params, opt_state = tup
        traj_batch, advantages, targets = batch_info

        def _loss_fn(params, traj_batch, gae, targets):
            # RERUN NETWORK
            logits, value = forward.apply(params, traj_batch.obs)
            pi = distrax.Categorical(logits=logits)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE VALUE LOSS
            value_loss = 0.5 * jnp.square(value - targets).mean()

            # CALCULATE ACTOR LOSS
            gae = reward_scaling(gae)
            loss_actor = -log_prob * gae
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(params, traj_batch, advantages, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), total_loss

    rng, _rng = jax.random.split(rng)
    batch_size = args.batch_size
    batch = (traj_batch, advantages, targets)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    (params, opt_state), loss_info = _update_batch((params, opt_state), batch)
    runner_state = (params, opt_state, env_state, last_obs, rng)
    return runner_state, loss_info


@jax.jit
def evaluate(params, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        logits, value = forward.apply(params, state.observation)
        # action = logits.argmax(axis=-1)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        return state, R + state.rewards, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    tt = 0
    st = time.time()
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    params = forward.init(_rng, init_x)
    opt_state = optimizer.init(params=params)

    # INIT UPDATE FUNCTION
    # _update_step = make_update_fn()
    jitted_roll_out = jax.jit(roll_out)
    jitted_calc_gae = jax.jit(calc_gae)
    jitted_update_step = jax.jit(uptate_step)
    # jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (params, opt_state, env_state, env_state.observation, _rng)

    # warm up
    # _, _ = jitted_update_step(runner_state)

    steps = 0

    # initial evaluation
    et = time.time()  # exclude evaluation time
    tt += et - st
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    wandb.log(log)
    st = time.time()

    for i in range(num_updates):
        runner_state, traj_batch = jitted_roll_out(runner_state)
        print(traj_batch)
        advantages, targes = jitted_calc_gae(runner_state, traj_batch)
        runner_state, loss_info = jitted_update_step(
            runner_state, traj_batch, advantages, targes
        )
        steps += args.num_envs * args.num_steps

        # evaluation
        et = time.time()  # exclude evaluation time
        tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(runner_state[0], _rng)
        log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)
        st = time.time()

    return runner_state


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)
