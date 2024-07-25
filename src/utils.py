import jax
import jax.numpy as jnp
import distrax

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def single_play_step_two_policy_commpetitive(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """
    # teamB_model_params = teamB_param

    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)
        rewards1 = state.rewards
        terminated1 = state.terminated
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===01==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        # actor teammate turn
        # print("===02==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated
        # print(f"actor team, action: {action}")
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===03==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def single_play_step_two_policy_commpetitive_deterministic(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """
    # teamB_model_params = teamB_param

    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]
    
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)
        rewards1 = state.rewards
        terminated1 = state.terminated
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===01==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        # actor teammate turn
        # print("===02==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated
        # print(f"actor team, action: {action}")
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===03==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def single_play_step_free_run(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """

    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)
        rewards1 = state.rewards
        terminated1 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated

        # actor teammate turn
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        batch_size = action.shape[0]
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, batch_size)
        state = jax.vmap(step_fn)(state, action, rngs)
        return state

    return wrapped_step_fn
