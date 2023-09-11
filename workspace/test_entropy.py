import jax
import jax.numpy as jnp
import distrax
<<<<<<< HEAD
import math
import numpy as np
from src.models import ActorCritic, make_forward_pass
import optax
import pgx.bridge_bidding as bb
import haiku as hk
import pickle


class Liner(hk.Module):
    def __init__(
        self,
        action_dim,
    ):
        super().__init__()
        self.action_dim = action_dim

    def __call__(self, x):
        logits = hk.Linear(self.action_dim)(x)
        return logits


def forward_fn(x):
    net = Liner(38)
    logits = net(x)
    return logits


forward_pass = hk.without_apply_rng(hk.transform(forward_fn))


def mul_exp(x: jnp.ndarray, logp: jnp.ndarray) -> jnp.ndarray:
    """Returns `x * exp(logp)` with zero output if `exp(logp)==0`.

    Args:
      x: An array.
      logp: An array.

    Returns:
      `x * exp(logp)` with zero output and zero gradient if `exp(logp)==0`,
      even if `x` is NaN or infinite.
    """
    p = jnp.exp(logp)
    # If p==0, the gradient with respect to logp is zero,
    # so we can replace the possibly non-finite `x` with zero.
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def masked_entropy(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Calculate the entropy for masked logits"""
    logits = logits + jnp.finfo(np.float64).min * (~mask)
    log_probs = jax.nn.log_softmax(logits)
    p_log_p = mul_exp(log_probs, log_probs)
    p_log_p = jnp.where(mask, p_log_p, 0)
    print(p_log_p)
    return -jnp.sum(p_log_p, axis=-1)


def miss_entropy(logits, mask):
    logits = logits + jnp.finfo(np.float64).min * (~mask)
    probs = jax.nn.softmax(logits)
    log_probs = jnp.log(jnp.clip(probs, 1e-35))
    entropy = -jnp.sum(log_probs * probs)
    return (
        -entropy,
        logits,
        probs,
        log_probs,
    )


def entropy_from_dif(logits, mask):
=======
import numpy as np
import pgx.bridge_bidding as bb
import haiku as hk


def entropy_from_dif(logits, mask):
    """非合法手をmaskしたエントロピーを定義から計算"""
>>>>>>> harukaki/feat/eval_log
    logits = logits + jnp.finfo(np.float64).min * (~mask)
    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    entropy = jnp.array(0, dtype=jnp.float32)
    for i in range(38):
        entropy = jax.lax.cond(
            mask[i], lambda: entropy + log_probs[i] * probs[i], lambda: entropy
        )
    return (
        -entropy,
        logits,
        probs,
        log_probs,
    )


def entropy_from_distrax(logits, mask):
<<<<<<< HEAD
=======
    """非合法手をmaskしたエントロピーをdistraxを用いて計算"""
>>>>>>> harukaki/feat/eval_log
    illegal_action_masked_logits = logits + jnp.finfo(np.float64).min * (~mask)
    illegal_action_masked_pi = distrax.Categorical(logits=illegal_action_masked_logits)
    return (
        illegal_action_masked_pi.entropy(),
        illegal_action_masked_logits,
        illegal_action_masked_pi.probs,
    )


<<<<<<< HEAD
@jax.jit
def loss_fn(logits, mask):
    entropy, _, _, _ = entropy_from_dif(logits, mask)
    return -entropy.mean()


"""
logits = jnp.array([[10, 1, 0, -2], [2, 1, 0, 5]], dtype=jnp.float32)
mask = jnp.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=jnp.bool_)
=======
logits = jnp.array([1, 1, 2, -2], dtype=jnp.float32)
mask = jnp.array([1, 1, 0, 1], dtype=jnp.bool_)
>>>>>>> harukaki/feat/eval_log

(
    entropy,
    logits,
    probs,
    log_probs,
<<<<<<< HEAD
) = jax.vmap(
    entropy_from_dif
)(logits, mask)
=======
) = entropy_from_dif(logits, mask)
print("calc from difinition")
>>>>>>> harukaki/feat/eval_log
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")

<<<<<<< HEAD
(
    entropy,
    logits,
    probs,
) = jax.vmap(
    entropy_from_distrax
)(logits, mask)
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")


true_grad = jax.value_and_grad(loss_fn)
value, grad = true_grad(logits, mask)
print(value)
print(grad)

new_logits = logits - grad
=======

@jax.jit
def loss_fn(logits, mask):
    entropy, _, _, _ = entropy_from_dif(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print(f"entropy: {value}")
print(f"grad: {grad}")


>>>>>>> harukaki/feat/eval_log
(
    entropy,
    logits,
    probs,
<<<<<<< HEAD
    log_probs,
) = jax.vmap(
    entropy_from_dif
)(new_logits, mask)
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")

logits = jnp.array([100, 1, 0, -2], dtype=jnp.float32)
pi = distrax.Categorical(logits)
print(pi.logits)
"""
=======
) = entropy_from_distrax(logits, mask)
print("calc from distrax")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _ = entropy_from_distrax(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print(f"entropy: {value}")
print(f"grad: {grad}")


# experiments in bridge bidding


class Liner(hk.Module):
    """1 Layer Liner"""

    def __init__(
        self,
        action_dim,
    ):
        super().__init__()
        self.action_dim = action_dim

    def __call__(self, x):
        logits = hk.Linear(self.action_dim)(x)
        return logits


def forward_fn(x):
    net = Liner(38)
    logits = net(x)
    return logits


forward_pass = hk.without_apply_rng(hk.transform(forward_fn))
>>>>>>> harukaki/feat/eval_log

env = bb.BridgeBidding()
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
state = env.init(_rng)
# INIT NETWORK
rng, _rng = jax.random.split(rng)
init_x = jnp.zeros((1,) + env.observation_shape)
params = forward_pass.init(_rng, init_x)  # params  # DONE
logits = forward_pass.apply(
    params,
    state.observation.astype(jnp.float32),
)
mask = state.legal_action_mask
<<<<<<< HEAD
print(f"logits: {logits}")
print(f"mask: {mask}")
=======
>>>>>>> harukaki/feat/eval_log

(
    entropy,
    logits,
    probs,
    log_probs,
) = entropy_from_dif(logits, mask)
<<<<<<< HEAD
=======
print("calc from difinition")
>>>>>>> harukaki/feat/eval_log
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")
<<<<<<< HEAD
print("\n")

(
    entropy,
    logits,
    probs,
) = entropy_from_distrax(logits, mask)
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")

true_grad = jax.value_and_grad(loss_fn)
value, grad = true_grad(logits, mask)
print(value)
print(grad)

new_logits = logits - grad
(
    entropy,
    logits,
    probs,
    log_probs,
) = entropy_from_dif(new_logits, mask)
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")
=======


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _, _ = entropy_from_dif(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print(f"entropy: {value}")
print(f"grad: {grad}")

>>>>>>> harukaki/feat/eval_log

(
    entropy,
    logits,
    probs,
<<<<<<< HEAD
    log_probs,
) = miss_entropy(logits, mask)
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")


@jax.jit
def miss_loss_fn(logits, mask):
    entropy, _, _, _ = miss_entropy(logits, mask)
    return -entropy.mean()


miss_grad = jax.value_and_grad(miss_loss_fn)
value, grad = miss_grad(logits, mask)
print(f"miss value: {value}")
print(f"miss grad: {grad}")
=======
) = entropy_from_distrax(logits, mask)
print("calc from distrax")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _ = entropy_from_distrax(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print(f"entropy: {value}")
print(f"grad: {grad}")
>>>>>>> harukaki/feat/eval_log
