import jax
import jax.numpy as jnp
import distrax

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    There are several concerns before staging this wrapper:

    1. Final state (observation)
    When auto restting happened, the termianl (or truncated) state/observation is replaced by initial state/observation,
    This is not problematic if it's termination.
    However, when truncation happened, value of truncated state/observation might be used by agent.
    So we have to preserve the final state (observation) somewhere.
    For example, in Gymnasium,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and terminates in reasonable # of steps.
    (NOTE: Chess, Shogi, and Go have `max_termination_steps` option following AlphaZero approach)
    So, curren implementation is enough (so far), and I feel implementing `final_state/observation` is useless and not necessary.

    2. Performance:
    Might harm the performance as it always generates new state.
    Memory usage might be doubled. Need to check.
    """

    def wrapped_step_fn(state, action):
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                _step_count=jnp.int32(0),
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn


def single_play_step_two_policy_commpetitive(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """
    # teamB_model_params = teamB_param

    def wrapped_step_fn(state, action, rng):
        state = jax.vmap(step_fn)(state, action)
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
        state = jax.vmap(step_fn)(state, action)  # step by left
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
        state = jax.vmap(step_fn)(state, action)  # step by pd
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
        state = jax.vmap(step_fn)(state, action)  # step by left
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
        state = jax.vmap(step_fn)(state, action)
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
        state = jax.vmap(step_fn)(state, action)  # step by left
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
        state = jax.vmap(step_fn)(state, action)  # step by pd
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
        state = jax.vmap(step_fn)(state, action)  # step by left
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
        state = jax.vmap(step_fn)(state, action)
        rewards1 = state.rewards
        terminated1 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        state = jax.vmap(step_fn)(state, action)  # step by left
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
        state = jax.vmap(step_fn)(state, action)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        state = jax.vmap(step_fn)(state, action)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        state = jax.vmap(step_fn)(state, action)
        return state

    return wrapped_step_fn
