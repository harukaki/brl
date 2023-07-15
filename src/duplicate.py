import jax
import jax.numpy as jnp
from typing import NamedTuple

import pgx
from pgx.bridge_bidding import BridgeBidding, State, _player_position, _observe
from pgx.experimental.utils import act_randomly

PASS_ACTION_NUM = 0
DOUBLE_ACTION_NUM = 1
REDOUBLE_ACTION_NUM = 2
BID_OFFSET_NUM = 3


def _imp_reward(
    table_a_reward: jnp.ndarray, table_b_reward: jnp.ndarray
) -> jnp.ndarray:
    """Convert score reward to IMP reward

    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 3.,  3., -3., -3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-3., -3.,  3.,  3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([-3500, -3500, 3500, 3500])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-23., -23.,  23.,  23.], dtype=float32)
    >>> table_a_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> table_b_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 24.,  24., -24., -24.], dtype=float32)
    """
    # fmt: off
    IMP_LIST = jnp.array([20, 50, 90, 130, 170,
                          220, 270, 320, 370, 430,
                          500, 600, 750, 900, 1100,
                          1300, 1500, 1750, 2000, 2250,
                          2500, 3000, 3500, 4000], dtype=jnp.float32)
    # fmt: on
    win = jax.lax.cond(
        table_a_reward[0] + table_b_reward[0] >= 0, lambda: 1, lambda: -1
    )

    def condition_fun(imp_diff):
        imp, difference_point = imp_diff
        return (difference_point >= IMP_LIST[imp]) & (imp < 24)

    def body_fun(imp_diff):
        imp, difference_point = imp_diff
        imp += 1
        return (imp, difference_point)

    imp, difference_point = jax.lax.while_loop(
        condition_fun,
        body_fun,
        (0, abs(table_a_reward[0] + table_b_reward[0])),
    )
    return jnp.array(
        [imp * win, imp * win, -imp * win, -imp * win], dtype=jnp.float32
    )


def _duplicate_init(
    state: State,
) -> State:
    """Make duplicated state where NSplayer and EWplayer are swapped

    >>> key = jax.random.PRNGKey(0)
    >>> state = env.init(key)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state._shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state._dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> state = env.step(state, 35)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state._shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state._dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> duplicate_state._pass_num
    Array(0, dtype=int32)

    >>> state = env.step(state, 0)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state._shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state._dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> duplicate_state.legal_action_mask
    Array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False, False], dtype=bool)
    """
    ix = jnp.array([1, 0, 3, 2])
    shuffled_players = state._shuffled_players[ix]
    current_player = shuffled_players[state._dealer]
    legal_actions = jnp.ones(38, dtype=jnp.bool_)
    # 最初はdable, redoubleできない
    legal_actions = legal_actions.at[DOUBLE_ACTION_NUM].set(False)
    legal_actions = legal_actions.at[REDOUBLE_ACTION_NUM].set(False)
    duplicated_state = State(  # type: ignore
        _shuffled_players=state._shuffled_players[ix],
        current_player=current_player,
        _hand=state._hand,
        _dealer=state._dealer,
        _vul_NS=state._vul_NS,
        _vul_EW=state._vul_EW,
        legal_action_mask=legal_actions,
    )
    return duplicated_state


def duplicate_init(state):
    state = _duplicate_init(state)
    obs = _observe(state, state.current_player)
    return state.replace(observation=jax.lax.stop_gradient(obs))


class Table_info(NamedTuple):
    terminated: jnp.ndarray
    rewards: jnp.ndarray
    last_bid: jnp.ndarray
    last_bidder: jnp.ndarray
    call_x: jnp.ndarray
    call_xx: jnp.ndarray


def duplicate_step(step_fn):
    def wrapped_step(state, action, table_a_info, table_b_info):
        state = step_fn(state, action)

        next_state = jax.lax.cond(
            ~table_a_info.terminated & state.terminated,
            lambda: duplicate_init(state),
            lambda: state,
        )

        next_state = jax.lax.cond(
            table_a_info.terminated
            & state.terminated
            & ~table_b_info.terminated,
            lambda: state.replace(  # type: ignore
                rewards=_imp_reward(table_a_info.rewards, state.rewards)
            ),
            lambda: next_state.replace(
                rewards=jnp.zeros(4, dtype=jnp.float32)
            ),
        )

        table_b_info = jax.lax.cond(
            state.terminated
            & table_a_info.terminated
            & ~table_b_info.terminated,
            lambda: Table_info(
                terminated=state.terminated,
                rewards=state.rewards,
                last_bid=state._last_bid,
                last_bidder=state._last_bidder,
                call_x=state._call_x,
                call_xx=state._call_xx,
            ),
            lambda: table_b_info,
        )
        table_a_info = jax.lax.cond(
            ~table_a_info.terminated & state.terminated,
            lambda: Table_info(
                terminated=state.terminated,
                rewards=state.rewards,
                last_bid=state._last_bid,
                last_bidder=state._last_bidder,
                call_x=state._call_x,
                call_xx=state._call_xx,
            ),
            lambda: table_a_info,
        )

        return next_state, table_a_info, table_b_info
        """
        return jax.lax.cond(
            ~state.terminated,
            lambda: (
                state,
                table_a_info,
                table_b_info,
            ),
            lambda: jax.lax.cond(
                table_a_info.terminated,
                lambda: jax.lax.cond(
                    ~table_b_info.terminated,
                    lambda: (
                        state.replace(  # type: ignore
                            rewards=_imp_reward(
                                table_a_info.rewards, state.rewards
                            )
                        ),
                        table_a_info,
                        Table_info(
                            terminated=state.terminated,
                            rewards=state.rewards,
                            last_bid=state._last_bid,
                            last_bidder=state._last_bidder,
                            call_x=state._call_x,
                            call_xx=state._call_xx,
                        ),
                    ),
                    lambda: (state, table_a_info, table_b_info),
                ),
                lambda: (
                    duplicate_init(state),
                    Table_info(
                        terminated=state.terminated,
                        rewards=state.rewards,
                        last_bid=state._last_bid,
                        last_bidder=state._last_bidder,
                        call_x=state._call_x,
                        call_xx=state._call_xx,
                    ),
                    table_b_info,
                ),
            ),
        )
        """

    return wrapped_step


if __name__ == "__main__":
    env = BridgeBidding()
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(duplicate_step(env.step)))
    env.step = jax.vmap(env.step)
    player_position = jax.vmap(_player_position)
    act_randomly = jax.jit(act_randomly)

    # show multi visualizations
    N = 4
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N)

    state: pgx.State = init(keys)
    # duplicate_state: pgx.State = duplicate_vmap(state)

    i = 0
    has_duplicate_result = jnp.zeros(N, dtype=jnp.bool_)
    table_a_info = Table_info(
        terminated=state.terminated,
        rewards=state.rewards,
        last_bid=state._last_bid,
        last_bidder=state._last_bidder,
        call_x=state._call_x,
        call_xx=state._call_xx,
    )
    table_b_info = Table_info(
        terminated=state.terminated,
        rewards=state.rewards,
        last_bid=state._last_bid,
        last_bidder=state._last_bidder,
        call_x=state._call_x,
        call_xx=state._call_xx,
    )
    print(table_a_info)
    print(table_b_info)
    while not state.terminated.all():
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state)

        print("================")
        print(f"{i:04d}")
        print("================")
        print(f"curr_player: {state.current_player}\naction: {action}")
        print(
            f"curr_player_position: {player_position(state.current_player, state)}"
        )
        print(f"shuflled_players:\n{state._shuffled_players}")
        state.save_svg(f"svg/{i:04d}.svg")
        state_check = env.step(state, action)
        state, table_a_info, table_b_info = step(
            state, action, table_a_info, table_b_info
        )
        print(state)
        print(table_a_info)
        print(table_b_info)
        # print(f"table a reward\n{table_a_reward}")
        # print(f"table b reward\n{table_b_reward}")
        print(f"score_reward:\n{state_check.rewards}")
        print(f"IMP_reward:\n{state.rewards}")
        # print(f"has_duplicate_result:\n{has_duplicate_result}")
        i += 1
    state.save_svg(f"svg/{i:04d}.svg")
