import jax.numpy as jnp
import jax


def make_calc_gae(config, actor_forward_pass):
    def calc_gae(runner_state, traj_batch):
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
        return advantages, targets

    return calc_gae
