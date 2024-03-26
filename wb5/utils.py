import numpy as np
import jax
import jax.numpy as jnp
from pgx.bridge_bidding import BridgeBidding, _state_to_pbn
from submodule.bridge_env.bridge_env import BiddingPhase, Vul, Player, Hands, Bid

def convert_act_pgx2be(action):
    bid_index = np.roll(np.arange(1, 39), -35)
    return Bid(bid_index[action])

def convert_leagal_action_mask(available_bid):
    leagal_action_mask = np.array(available_bid)
    return np.roll(leagal_action_mask, -35)

def convert_vul(active_player, vul):
    return np.array([not active_player.pair.is_vul(vul), active_player.pair.is_vul(vul), not active_player.opponent_pair.is_vul(vul), active_player.opponent_pair.is_vul(vul)], dtype=np.bool_)

def convert_card(card):
    return (card % 13) * 4 + card // 13 

def convert_hand(hand):
    pgx_hand = np.zeros(52, dtype=np.bool_)
    for i in range(52):
        if hand[i] == True:
            pgx_hand[convert_card(i)] = True
    return pgx_hand
    
def convert_history(dealer, active_player, bid_history):
    last_bid = 0
    obs_history = np.zeros(424, dtype=np.bool_)
    dealer = dealer.value - 1
    active_player = active_player.value - 1
    for i, bid in enumerate(bid_history):
        relative_bidder = ((i + dealer) % 4 + (4 - active_player)) % 4
        bid_int = bid.value
        if bid_int <= 35:
            last_bid = bid_int
            obs_history[4 + (bid_int - 1) * 4 * 3 + relative_bidder] = True
        elif bid_int == 36:
            if last_bid == 0:
                obs_history[relative_bidder] = True
        elif bid_int == 37:
            obs_history[4 + (last_bid - 1) * 4 * 3 + 4 + relative_bidder] = True
        elif bid_int == 38:
            obs_history[4 + (last_bid - 1) * 4 * 3 + 4 * 2 + relative_bidder] = True
    return obs_history

def convert_obs(dealer, vul, active_player, bid_history, hand):
    obs_vul = convert_vul(active_player=active_player, vul=vul)
    obs_history = convert_history(dealer=dealer, active_player=active_player, bid_history=bid_history)
    obs_hand = convert_hand(hand=hand)
    return np.concatenate((obs_vul, obs_history, obs_hand))

if __name__ == "__main__":
    env = BridgeBidding()
    init = env.init
    step = env.step
    key = jax.random.PRNGKey(99)
    state = init(key)
    bp = BiddingPhase(dealer= Player(2),vul=Vul(1))
    pgx_bids = [0, 9, 11, 20, 
               1, 0, 22, 1,
               2, 0, 0, 28,
               0, 0]
    bids = [Bid.Pass, Bid.D2, Bid.S2, Bid.H4,
                    Bid.X, Bid.Pass, Bid.NT4, Bid.X,
                    Bid.XX, Bid.Pass, Bid.Pass, Bid.C6,
                    Bid.Pass, Bid.Pass]
    state = state.replace(
        _dealer=jnp.int32(1),
        current_player=jnp.int32(3),
        _shuffled_players=jnp.array([0, 3, 1, 2], dtype=jnp.int32),
        _vul_NS=jnp.bool_(0),
        _vul_EW=jnp.bool_(0),
    )
    pbn = _state_to_pbn(state)
    hands = Hands.convert_pbn(pbn)
    print(f"pgx hand: {pbn}")
    print(f"bridge_env hand: {hands.to_pbn()}")
    for i in range(14):
            bp_bid = convert_act_pgx2be(pgx_bids[i])
            print(bp_bid)
            bp.take_bid(bp_bid)
            state = step(state, pgx_bids[i])
            state.save_svg("test.svg")
            pgx_obs = state.observation
            be_obs = convert_obs(dealer=bp.dealer, vul=bp.vul, active_player=bp.active_player, bid_history=bp.bid_history, hand=hands.to_binary()[bp.active_player])
            assert np.all(pgx_obs == be_obs)
            assert np.all(state.legal_action_mask == convert_leagal_action_mask(bp.available_bid))