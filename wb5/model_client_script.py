import argparse
import logging
from typing import Tuple
import pickle
from submodule.bridge_env.bridge_env import Bid, BiddingPhase, Player,Vul
from submodule.bridge_env.bridge_env.network_bridge import Client
from submodule.bridge_env.bridge_env.network_bridge.bidding_system import BiddingSystem
from submodule.bridge_env.bridge_env.network_bridge.playing_system import RandomPlay

from .utils import convert_leagal_action_mask, convert_vul, convert_hand, convert_history, convert_act_pgx2be, convert_obs
from .models import make_forward_pass
import jax.numpy as jnp
import distrax
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple





# TODO: Consider to reduce one of bid_history and players_bid_history.
#  Introduce dealer.
@dataclass(frozen=True)
class State:
    """Bidding State (active_player, active_player's hand, available_bid, vul,
    bid_history, players_bid_history)

    :param active_player: player who takes an action
    :param player_hand: 52dims binary vector, (element: 0 or 1, tuple)
    :param vul: vulnerable
    :param available_bid: 38dims binary vector, (element: 0 or 1, tuple)
    :param bid_history: list of Bid
    :param players_bid_history: dict {Player.N: list of Bid, Player.E: list,
        Player.S: list, Player.W: list}
    """
    active_player: Player
    player_hand: Tuple[int, ...]
    available_bid: Tuple[int, ...]
    vul: Vul
    bid_history: Tuple[Bid, ...]
    players_bid_history: Dict[Player, List[Bid]]

    @classmethod
    def env_to_state(cls,
                     hand: Tuple[int, ...],
                     env: BiddingPhase):
        """Makes State instance from BiddingPhase environment.

        :param hand: player hand.
        :param env: BiddingPhase environment.
        :return: State instance.
        """
        assert env.active_player is not None, 'Bidding phase has already done.'

        return State(player_hand=hand,
                     vul=env.vul,
                     active_player=env.active_player,
                     available_bid=tuple(env.available_bid),
                     bid_history=tuple(env.bid_history),
                     players_bid_history=copy.deepcopy(env.players_bid_history)
                     )



class ModelBiddingSystem(BiddingSystem):
    def __init__(self,
                 model_type = "DeepMind",
                 activation = "relu",
                 model_path = "models/sl/params-400000.pkl"):

        self.model_params = pickle.load(open(model_path, "rb"))
        self.forward_pass = make_forward_pass(activation, model_type)

    def bid(self, hand: Tuple[int, ...], bidding_phase: BiddingPhase) -> Bid:
        state = State.env_to_state(hand, bidding_phase)
        print(state)
        obs = convert_obs(dealer=bidding_phase.dealer, vul=bidding_phase.vul, active_player=bidding_phase.active_player, bid_history=bidding_phase.bid_history, hand=hand)
        legal_action_mask = convert_leagal_action_mask(state.available_bid)

        logits, _ = self.forward_pass.apply(self.model_params, obs)
        logits = logits + jnp.finfo(jnp.float64).min * (
            ~legal_action_mask.astype(jnp.bool_)
        )
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        bid = convert_act_pgx2be(action)
        print(bid)
        return bid

    


if __name__ == '__main__':
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        default=2000,
                        type=int,
                        help='Port number. (default=2000)')
    parser.add_argument('-i', '--ip_address',
                        default='localhost',
                        type=str,
                        help='IP address. (default=localhost)')
    parser.add_argument('-l', '--location',
                        default='N',
                        type=str,
                        help='Player (N, E, S or W)')
    parser.add_argument('-t', '--team_name',
                        default='teamNS',
                        type=str,
                        help='Team name')
    parser.add_argument('-a', '--activation',
                        default='Activation',
                        type=str,
                        help='Activation')
    parser.add_argument('-mt', '--model_type',
                        default='FAIR',
                        type=str,
                        help='Model type. (FAIR)')
    parser.add_argument('-m', '--model_path',
                        type=str,
                        help='Model file path.')

    args = parser.parse_args()
    player = Player[args.location]
    with Client(player=player,
                team_name=str(args.team_name),
                bidding_system=ModelBiddingSystem(model_path=args.model_path,
                                                  model_type=args.model_type,
                                                  activation=args.activation),
                playing_system=RandomPlay(),
                ip_address=args.ip_address,
                port=args.port) as client:
        print(client)
        client.run()
        print('end')
