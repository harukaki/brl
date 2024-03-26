from submodule.bridge_env.bridge_env.data_handler.json_handler.parser import JsonParser
from submodule.bridge_env import Hands, Player
from pgx.bridge_bidding import _key_to_hand, _card_str_to_int, _to_binary, BridgeBidding 
import jax.numpy as jnp
import jax
import numpy as np


env = BridgeBidding()
init = jax.jit(env.init)
step = jax.jit(env.step)

def _pbn_to_key(pbn: str):
    """Convert pbn to key of dds table"""
    key = jnp.zeros(52, dtype=jnp.int32)
    hands = pbn[2:]
    for player, hand in enumerate(list(hands.split())):  # for each player
        for suit, cards in enumerate(list(hand.split("."))):  # for each suit
            for card in cards:  # for each card
                card_num = _card_str_to_int(card) + suit * 13
                key = key.at[card_num].set(player)
    key = key.reshape(4, 13)
    return _to_binary(key)

def convert_hands(hands):
    return _key_to_hand(_pbn_to_key(hands.to_pbn()))


def convert_vul(vul):
    vul_NS = jnp.bool_(False)
    vul_EW = jnp.bool_(False)
    if vul.value in (2, 4):
        vul_NS = jnp.bool_(True)
    if vul.value in (3, 4):
        vul_EW = jnp.bool_(True)
    return vul_NS, vul_EW

def convert_dealer(dealer):
    return dealer.value - 1

def convert_bid(bid):
    bid_idx = np.roll(np.arange(38), -3)
    return bid_idx[bid.value - 1]

def vis_boardlog_pgx(board_log, fig_path):
    hands =  convert_hands(board_log.hands)
    vul_NS, vul_EW = convert_vul(board_log.vul)
    dealer = convert_dealer(board_log.dealer)
    key = jax.random.PRNGKey(99)
    state = init(key)
    state = state.replace(
    _hand=hands,
    _dealer=dealer,
    _vul_NS=vul_NS,
    _vul_EW=vul_EW,
    )
    for bid in board_log.bid_history:
        action = convert_bid(bid)
        state = step(state, action)
    state.save_svg(fig_path)


if __name__ == "__main__":
    with open("test_network/board_log/table1_board_0000.json", mode='r') as f:
        data = JsonParser().parse_board_logs(f)

    vis_boardlog_pgx(data[0], "test.svg")