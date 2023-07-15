import haiku as hk
import jax
import jax.numpy as jnp


class ActorCritic(hk.Module):
    def __init__(
        self,
        action_dim,
        activation="relu",
        model="FAIR",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.activation = activation
        self.model = model

    def __call__(self, x):
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        if self.model == "DeepMind":
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            actor_mean = hk.Linear(self.action_dim)(x)
            critic = hk.Linear(1)(x)
        elif self.model == "FAIR":
            input = x
            x = hk.Linear(200)(x)
            shortcut_1 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_1
            shortcut_2 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_2
            x = hk.Linear(200)(x)
            x = jnp.concatenate([x, input], axis=-1)
            x = hk.Linear(200)(x)
            shortcut_3 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_3
            shortcut_4 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_4
            actor_mean = hk.Linear(self.action_dim)(x)
            critic = hk.Linear(1)(x)

        return actor_mean, jnp.squeeze(critic, axis=-1)


def DeepMind_sl_net_fn(x):
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


def make_forward_pass(activation, model_type):
    def forward_fn(x):
        net = ActorCritic(
            38,
            activation=activation,
            model=model_type,
        )
        logits, value = net(x)
        return logits, value

    return hk.without_apply_rng(hk.transform(forward_fn))
