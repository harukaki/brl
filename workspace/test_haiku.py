import haiku as hk
import jax.numpy as jnp
import jax


def teat(x):
    return jax.lax.cond(x == 0, lambda: 0, lambda: -a)


print(teat(1))


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(images, labels):
    mlp = hk.Sequential(
        [
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    logits = mlp(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))


loss_fn_t = hk.transform(loss_fn)
loss_fn_t = hk.without_apply_rng(loss_fn_t)

rng = jax.random.PRNGKey(42)
init_x = jnp.zeros((1,) + jnp.zeros(10))
print(init_x)
params = loss_fn_t.init(rng, init_x)
print(params)


def update_rule(param, update):
    return param - 0.01 * update


for images, labels in input_dataset:
    grads = jax.grad(loss_fn_t.apply)(params, images, labels)
    params = jax.tree_util.tree_map(update_rule, params, grads)
