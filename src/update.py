import jax
import jax.numpy as jnp
import distrax
import numpy as np
import optax
from .models import make_forward_pass


def make_update_step(config, actor_forward_pass, optimizer):
    def make_policy(config):
        if config["actor_illegal_action_mask"]:

            def masked_policy(mask, logits):
                logits = logits + jnp.finfo(np.float64).min * (~mask)
                pi = distrax.Categorical(logits=logits)
                return pi

            return masked_policy
        elif config["actor_illegal_action_penalty"]:

            def no_masked_policy(mask, logits):
                pi = distrax.Categorical(logits=logits)
                return pi

            return no_masked_policy

    policy = make_policy(config)

    def _get(x, i):
        return x[i]

    def make_reward_scaling(config):
        if config["reward_scaling"]:

            def reward_scaling(gae):
                return (gae - gae.mean()) / (gae.std() + 1e-8)

            return reward_scaling

        else:

            def no_scaling(gae):
                return gae

            return no_scaling

    reward_scaling = make_reward_scaling(config)

    def make_value_loss_fn(config):
        if config["value_clipping"]:

            def clipped_value_loss_fn(value, traj_batch, targets):
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                    -config["clip_eps"], config["clip_eps"]
                )
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )
                return value_loss

            return clipped_value_loss_fn
        else:

            def value_loss_fn(value, traj_batch, targets):
                value_loss = 0.5 * jnp.square(value - targets).mean()
                return value_loss

            return value_loss_fn

    value_loss_fn = make_value_loss_fn(config)

    def update_step(runner_state, traj_batch, advantages, targets):
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = actor_forward_pass.apply(
                        params, traj_batch.obs.astype(jnp.float32)
                    )  # DONE
                    mask = traj_batch.legal_action_mask
                    pi = policy(mask, logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_loss = value_loss_fn(
                        value=value, traj_batch=traj_batch, targets=targets
                    )
                    """
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["clip_eps"], config["clip_eps"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    """

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)

                    # gae標準化
                    gae = reward_scaling(gae)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["clip_eps"],
                            1.0 + config["clip_eps"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    illegal_action_masked_logits = logits + jnp.finfo(
                        np.float64
                    ).min * (~mask)
                    illegal_action_masked_pi = distrax.Categorical(
                        logits=illegal_action_masked_logits
                    )
                    entropy = illegal_action_masked_pi.entropy().mean()

                    pi = distrax.Categorical(logits=logits)
                    illegal_action_probabilities = pi.probs * ~mask
                    illegal_action_loss = (
                        jnp.linalg.norm(illegal_action_probabilities, ord=2) / 2
                    )

                    total_loss = (
                        loss_actor
                        + config["vf_coef"] * value_loss
                        - config["ent_coef"] * entropy
                        + config["illegal_action_l2norm_coef"] * illegal_action_loss
                    )
                    """
                    total_loss = -config["ent_coef"] * entropy
                    """
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipflacs = jnp.float32(
                        jnp.abs((ratio - 1.0)) > config["clip_eps"]
                    ).mean()

                    return total_loss, (
                        value_loss,
                        loss_actor,
                        entropy,
                        approx_kl,
                        clipflacs,
                        illegal_action_loss,
                    )

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    params, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)  # DONE
                return (
                    params,
                    opt_state,
                ), total_loss  # DONE

            (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state  # DONE
            rng, _rng = jax.random.split(rng)
            batch_size = config["minibatch_size"] * config["num_minibatches"]
            assert (
                batch_size == config["num_steps"] * config["num_envs"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["num_minibatches"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )  # DONE
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )  # DONE
            return update_state, total_loss

        update_state = (
            params,
            opt_state,
            traj_batch,
            advantages,
            targets,
            rng,
        )  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["update_epochs"]
        )
        # print(loss_info)
        params, opt_state, _, _, _, rng = update_state  # DONE

        runner_state = (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        )  # DONE
        return runner_state, loss_info

    return update_step
