import jax
import jax.numpy as jnp
import distrax
import numpy as np
import optax
from models import make_forward_pass


def make_update_step(config, actor_forward_pass, optimizer):
    def make_policy(config):
        if config["ACTOR_ILLEGAL_ACTION_MASK"]:

            def masked_policy(mask, logits):
                logits = logits + jnp.finfo(np.float64).min * (~mask)
                pi = distrax.Categorical(logits=logits)
                return pi

            return masked_policy
        elif config["ACTOR_ILLEGAL_ACTION_PENALTY"]:

            def no_masked_policy(mask, logits):
                pi = distrax.Categorical(logits=logits)
                return pi

            return no_masked_policy

    policy = make_policy(config)

    def _get(x, i):
        return x[i]

    def make_reward_scaling(config):
        if config["REWARD_SCALING"]:

            def reward_scaling(gae):
                return (gae - gae.mean()) / (gae.std() + 1e-8)

            return reward_scaling

        else:

            def no_scaling(gae):
                return gae

            return no_scaling

    reward_scaling = make_reward_scaling(config)

    def make_value_loss_fn(config):
        if config["VALUE_CLIPPING"]:

            def clipped_value_loss_fn(value, traj_batch, targets):
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                    -config["CLIP_EPS"], config["CLIP_EPS"]
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
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
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
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
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
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                        + config["ILLEGAL_ACTION_L2NORM_COEF"] * illegal_action_loss
                    )
                    """
                    total_loss = -config["ENT_COEF"] * entropy
                    """
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipflacs = jnp.float32(
                        jnp.abs((ratio - 1.0)) > config["CLIP_EPS"]
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
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
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
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
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
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
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


def make_update_fn(config, env_step_fn, env_init_fn):
    actor_forward_pass = make_forward_pass(
        activation=config["ACTOR_ACTIVATION"],
        model_type=config["ACTOR_MODEL_TYPE"],
    )

    def make_policy(config):
        if config["ACTOR_ILLEGAL_ACTION_MASK"]:

            def masked_policy(mask, logits):
                logits = logits + jnp.finfo(np.float64).min * (~mask)
                pi = distrax.Categorical(logits=logits)
                return pi

            return masked_policy
        elif config["ACTOR_ILLEGAL_ACTION_PENALTY"]:

            def no_masked_policy(mask, logits):
                pi = distrax.Categorical(logits=logits)
                return pi

            return no_masked_policy

    policy = make_policy(config)

    if config["GAME_MODE"] == "competitive":
        if config["SELF_PLAY"]:

            def self_play_step_fn(
                step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
            ):
                return single_play_step_two_policy_commpetitive(
                    step_fn=step_fn,
                    actor_forward_pass=actor_forward_pass,
                    actor_params=actor_params,
                    opp_forward_pass=actor_forward_pass,
                    opp_params=actor_params,
                )

            make_step_fn = self_play_step_fn
        else:
            make_step_fn = single_play_step_two_policy_commpetitive
        opp_forward_pass = make_forward_pass(
            activation=config["OPP_ACTIVATION"],
            model_type=config["OPP_MODEL_TYPE"],
        )
        opp_params = pickle.load(open(config["OPP_MODEL_PATH"], "rb"))
    elif config["GAME_MODE"] == "free-run":
        make_step_fn = single_play_step_free_run
        opp_forward_pass = None
        opp_params = None

    def make_reward_scaling(config):
        if config["REWARD_SCALING"]:

            def reward_scaling(gae):
                return (gae - gae.mean()) / (gae.std() + 1e-8)

            return reward_scaling

        else:

            def no_scaling(gae):
                return gae

            return no_scaling

    reward_scaling = make_reward_scaling(config)

    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES

        # step_fn = _make_step(config["ENV_NAME"], runner_state[0])  # DONE
        step_fn = make_step_fn(
            step_fn=auto_reset(env_step_fn, env_init_fn),
            actor_forward_pass=actor_forward_pass,
            actor_params=runner_state[0],
            opp_forward_pass=opp_forward_pass,
            opp_params=opp_params,
        )
        get_fn = _get

        def _env_step(runner_state, unused):
            (
                params,
                opt_state,
                env_state,
                last_obs,
                terminated_count,
                rng,
            ) = runner_state  # DONE
            actor = env_state.current_player
            logits, value = actor_forward_pass.apply(
                params,
                last_obs.astype(jnp.float32),
            )  # DONE
            rng, _rng = jax.random.split(rng)
            mask = env_state.legal_action_mask
            pi = policy(mask, logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state = step_fn(env_state, action, _rng)
            terminated_count += jnp.sum(env_state.terminated)
            transition = Transition(
                env_state.terminated,
                action,
                value,
                jax.vmap(get_fn)(env_state.rewards / config["REWARD_SCALE"], actor),
                log_prob,
                last_obs,
                mask,
            )
            runner_state = (
                params,
                opt_state,
                env_state,
                env_state.observation,
                terminated_count,
                rng,
            )  # DONE
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )
        # print(traj_batch)

        # CALCULATE ADVANTAGE
        (
            params,
            opt_state,
            env_state,
            last_obs,
            terminated_count,
            rng,
        ) = runner_state  # DONE
        _, last_val = actor_forward_pass.apply(
            params, last_obs.astype(jnp.float32)
        )  # DONE

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            """
            def scan(f, init, xs, length=None):
                if xs is None:
                    xs = [None] * length
                carry = init
                ys = []
                for x in xs:
                    carry, y = f(carry, x)
                    ys.append(y)
                return carry, ys

            _, advantages = scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
            )
            """

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # print(traj_batch)
        # print(last_val)

        advantages, targets = _calculate_gae(traj_batch, last_val)
        # print(advantages)
        # print(targets)

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
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)

                    # gae標準化
                    gae = reward_scaling(gae)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
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
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                        + config["ILLEGAL_ACTION_L2NORM_COEF"] * illegal_action_loss
                    )
                    """
                    total_loss = -config["ENT_COEF"] * entropy
                    """
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipflacs = jnp.float32(
                        jnp.abs((ratio - 1.0)) > config["CLIP_EPS"]
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
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
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
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
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
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
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

    return _update_step
