import pyspiel
import jax
import jax.numpy as jnp
import numpy as np
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import (
    BridgeBidding,
    _state_to_pbn,
    _convert_card_pgx_to_openspiel,
)
import os

# from src.models import make_forward_pass
import pickle
import distrax

"""
train_dds_results_list = sorted(
    [
        path
        for path in os.listdir("dds_results")
        if ("train" in path) or ("test" in path)
    ]
)
"""
N = 3

pgx_env = BridgeBidding("100_hash.npy")
init = jax.jit((pgx_env.init))
step = jax.jit((pgx_env.step))
act_randomly = jax.jit(act_randomly)
"""
print(train_dds_results_list)

sl_forward_pass = make_forward_pass(
    activation="relu",
    model_type="DeepMind",
)
sl_params = pickle.load(open("sl_params/params-290000.pkl", "rb"))


for file in train_dds_results_list:
    pgx_env = BridgeBidding(os.path.join("dds_results", file))
    init = jax.jit((pgx_env.init))
    step = jax.jit((pgx_env.step))
    key = jax.random.PRNGKey(0)
    for k in range(N - 1):
        key, subkey = jax.random.split(key)
        pgx_state: pgx.State = init(subkey)
        # init vul
        if pgx_state._dealer == 0 or pgx_state._dealer == 2:
            is_dealer_vul = pgx_state._vul_NS
            is_non_dealer_vul = pgx_state._vul_EW
        else:
            is_dealer_vul = pgx_state._vul_EW
            is_non_dealer_vul = pgx_state._vul_NS

        if is_dealer_vul:
            if is_non_dealer_vul:
                openspiel_env = pyspiel.load_game(
                    "bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=true)"
                )
            else:
                openspiel_env = pyspiel.load_game(
                    "bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=false)"
                )
        else:
            if is_non_dealer_vul:
                openspiel_env = pyspiel.load_game(
                    "bridge(use_double_dummy_result=true,dealer_vul=false,non_dealer_vul=true)"
                )
            else:
                openspiel_env = pyspiel.load_game(
                    "bridge(use_double_dummy_result=true,dealer_vul=false,non_dealer_vul=false)"
                )

        # init hand
        openspiel_state = openspiel_env.new_initial_state()
        hand = pgx_state._hand.reshape(4, 13)
        hand = jnp.roll(hand, -pgx_state._dealer, axis=0)
        actions = []
        for i in range(13):
            for j in range(4):
                actions.append(_convert_card_pgx_to_openspiel(hand[j][i]))
        for action in actions:
            openspiel_state.apply_action(action)
        i = 0
        print("================")
        print(f"game: {k:04d}")
        while not pgx_state.terminated.all():
            key, subkey = jax.random.split(key)
            # action = act_randomly(subkey, pgx_state)
            logits_old, value = sl_forward_pass.apply(
                sl_params, pgx_state.observation
            )  # DONE
            mask_logits = jnp.finfo(np.float64).min * (~pgx_state.legal_action_mask)
            logits = logits_old + mask_logits
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=subkey)
            assert jnp.all(
                jnp.array(openspiel_state.observation_tensor()[4:484])
                == pgx_state.observation
            )
            # print("obs: ok")
            arr = np.zeros(38, dtype=bool)
            for value in openspiel_state.legal_actions():
                index = value - 52
                arr[index] = True
            # print(pgx_state.legal_action_mask)
            # print(jnp.array(arr))
            # print(f"pgx reward:\n{pgx_state.rewards}")
            # print(f"openspiel reward:\n{openspiel_state.rewards()}")
            assert jnp.all(pgx_state.legal_action_mask == jnp.array(arr))
            # print("action mask: ok")
            pgx_state = step(pgx_state, action)
            openspiel_state.apply_action(action + 52)
            assert jnp.all(
                pgx_state.rewards[
                    jnp.roll(pgx_state._shuffled_players, -pgx_state._dealer)
                ]
                == jnp.array(openspiel_state.rewards())
            )
            if pgx_state.terminated.all():
                print("obs: ok")
                print("action mask: ok")
                print("rewards: ok")
                print(openspiel_state)

            i += 1

"""
key = jax.random.PRNGKey(0)
# visualize last case
key, subkey = jax.random.split(key)
pgx_state: pgx.State = init(subkey)
# init vul
if pgx_state._dealer == 0 or pgx_state._dealer == 2:
    is_dealer_vul = pgx_state._vul_NS
    is_non_dealer_vul = pgx_state._vul_EW
else:
    is_dealer_vul = pgx_state._vul_EW
    is_non_dealer_vul = pgx_state._vul_NS

if is_dealer_vul:
    if is_non_dealer_vul:
        openspiel_env = pyspiel.load_game(
            "bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=true)"
        )
    else:
        openspiel_env = pyspiel.load_game(
            "bridge(use_double_dummy_result=true,dealer_vul=true,non_dealer_vul=false)"
        )
else:
    if is_non_dealer_vul:
        openspiel_env = pyspiel.load_game(
            "bridge(use_double_dummy_result=true,dealer_vul=false,non_dealer_vul=true)"
        )
    else:
        openspiel_env = pyspiel.load_game(
            "bridge(use_double_dummy_result=trues,dealer_vul=false,non_dealer_vul=false)"
        )

# init hand

openspiel_state = openspiel_env.new_initial_state()
hand = pgx_state._hand.reshape(4, 13)
hand = jnp.roll(hand, -pgx_state._dealer, axis=0)
actions = []
for i in range(13):
    for j in range(4):
        actions.append(_convert_card_pgx_to_openspiel(hand[j][i]))
for action in actions:
    openspiel_state.apply_action(action)
i = 0
while not pgx_state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, pgx_state)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"pgx curr_player: {pgx_state.current_player}\npgx action: {action}")
    pgx_state.save_svg(f"svg/{i:04d}.svg")
    print(openspiel_state)
    valid = jnp.all(
        jnp.array(openspiel_state.observation_tensor()[4:484]) == pgx_state.observation
    )
    print(f"spiel obs: {openspiel_state.observation_tensor()[4:484]}")
    print(f"pgx obs: {pgx_state.observation}")
    print(f"obs valid: {valid}")
    if not valid:
        print(
            f"obs vul valid: {jnp.all(jnp.array(openspiel_state.observation_tensor()[4:8]) == pgx_state.observation[:4])}"
        )
        print(
            f"obs hand valid: {jnp.all(jnp.array(openspiel_state.observation_tensor()[432:484])== pgx_state.observation[428:480])}"
        )
        print(
            f"obs history valid: {jnp.all(jnp.array(openspiel_state.observation_tensor()[8:432])== pgx_state.observation[4:428])}"
        )
    pgx_state = step(pgx_state, action)
    openspiel_state.apply_action(action + 52)

    print(f"pgx reward:\n{pgx_state.rewards}")
    print(f"openspiel reward:\n{openspiel_state.rewards()}")
    print(
        f"rewards valid: {jnp.all(pgx_state.rewards[jnp.roll(pgx_state._shuffled_players, -pgx_state._dealer)] == jnp.array(openspiel_state.rewards()))}"
    )
    i += 1
