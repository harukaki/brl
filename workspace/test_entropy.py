import jax
import jax.numpy as jnp
import distrax
import numpy as np
import pgx.bridge_bidding as bb
import haiku as hk


def entropy_from_dif(logits, mask):
    """非合法手をmaskしたエントロピーを定義から計算"""
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
    """非合法手をmaskしたエントロピーをdistraxを用いて計算"""
    illegal_action_masked_logits = logits + jnp.finfo(np.float64).min * (~mask)
    illegal_action_masked_pi = distrax.Categorical(logits=illegal_action_masked_logits)
    return (
        illegal_action_masked_pi.entropy(),
        illegal_action_masked_logits,
        illegal_action_masked_pi.probs,
    )


logits = jnp.array([1, 1, 2, -2], dtype=jnp.float32)
mask = jnp.array([1, 1, 0, 1], dtype=jnp.bool_)

(
    entropy,
    logits,
    probs,
    log_probs,
) = entropy_from_dif(logits, mask)
print("Calc from difinition")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")
print("\n")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _, _ = entropy_from_dif(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print(f"entropy: {value}")
print(f"grad: {grad}")
print("\n")


(
    entropy,
    logits,
    probs,
) = entropy_from_distrax(logits, mask)
print("Calc from distrax")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _ = entropy_from_distrax(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print("\n")
print(f"entropy: {value}")
print(f"grad: {grad}")
print("\n")

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

(
    entropy,
    logits,
    probs,
    log_probs,
) = entropy_from_dif(logits, mask)
print("Calc from difinition")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")
print(f"log_probs: {log_probs}")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _, _ = entropy_from_dif(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print("\n")
print(f"entropy: {value}")
print(f"grad: {grad}")
print("\n")

(
    entropy,
    logits,
    probs,
) = entropy_from_distrax(logits, mask)
print("Calc from distrax")
print(f"entropy: {entropy}")
print(f"logits: {logits}")
print(f"probs: {probs}")


@jax.jit
def loss_fn(logits, mask):
    entropy, _, _ = entropy_from_distrax(logits, mask)
    return -entropy.mean()


grad_fn = jax.value_and_grad(loss_fn)
value, grad = grad_fn(logits, mask)
print("\n")
print(f"entropy: {value}")
print(f"grad: {grad}")
