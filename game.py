import jax
import jax.numpy as jnp
import pickle
import distrax
from pgx.bridge_bidding import BridgeBidding
from src.models import make_forward_pass

# Load AI Model
robot_model_path = "bridge_models/model-pretrained-rl.pkl"  # Replace with actual path
robot_params = pickle.load(open(robot_model_path, "rb"))

# Define AI model architecture
robot_forward_pass = make_forward_pass(activation="relu", model_type="DeepMind")

# Initialize the environment
eval_env = BridgeBidding("dds_results/dds_results_10M.npy")

# Define bid mappings (Official NESW order)
BID_LABELS = [
    "PASS",
    "DOUBLE",
    "REDOUBLE",
    "1C",
    "1D",
    "1H",
    "1S",
    "1N",
    "2C",
    "2D",
    "2H",
    "2S",
    "2N",
    "3C",
    "3D",
    "3H",
    "3S",
    "3N",
    "4C",
    "4D",
    "4H",
    "4S",
    "4N",
    "5C",
    "5D",
    "5H",
    "5S",
    "5N",
    "6C",
    "6D",
    "6H",
    "6S",
    "6N",
    "7C",
    "7D",
    "7H",
    "7S",
    "7N",
]

# Seating order and mapping
SEATS = ["North", "East", "South", "West"]
player_seat_map = {i: seat for i, seat in enumerate(SEATS)}

# Configure which players are human or AI
player_config = {
    "North": True,  # Change to False for AI
    "East": False,
    "South": False,
    "West": False,
}


def get_vulnerability(state):
    """Convert vulnerability flags into human-readable format."""
    ns_vul = state._vul_NS.item()
    ew_vul = state._vul_EW.item()
    if ns_vul and ew_vul:
        return "Both"
    elif ns_vul:
        return "NS"
    elif ew_vul:
        return "EW"
    else:
        return "None"


def get_hand(state, player_id):
    """Extract and display only the 13 cards for the current player."""
    all_cards = state._hand  # Full deck (52 cards)
    player_positions = state._shuffled_players  # NESW assignment

    # Get the 13 cards for this player
    player_index = jnp.where(player_positions == player_id)[0].item()
    player_hand = all_cards[player_index * 13 : (player_index + 1) * 13]

    suits = ["♠", "♥", "♦", "♣"]
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

    # Organize cards into suits
    hand_by_suit = {suit: [] for suit in suits}
    for card_idx in player_hand:
        suit = suits[card_idx // 13]
        rank = ranks[card_idx % 13]
        hand_by_suit[suit].append(rank)

    return hand_by_suit


def parse_state(state, current_player, seat_name):
    """Display the current bidding state, hand, and vulnerability."""
    print("\n----- Current Game State -----")

    # Vulnerability
    vulnerability = get_vulnerability(state)
    print(f"Vulnerability: {vulnerability}")

    # Bidding history (Chronological)
    bid_log = []
    for i, action in enumerate(state._bidding_history):
        if action == -1:  # No more bids recorded
            break
        bid_log.append((i % 4, BID_LABELS[action]))

    print("Bidding History:")
    for player, bid in bid_log:
        print(f"  {player_seat_map[player]} → {bid}")

    # Display player's hand
    hand_by_suit = get_hand(state, current_player)
    print(f"\nYour Hand ({seat_name}):")
    for suit in ["♠", "♥", "♦", "♣"]:  # Maintain suit order
        print(
            f"  {suit}: {' '.join(hand_by_suit[suit]) if hand_by_suit[suit] else '—'}"
        )

    print("------------------------------\n")


def predict_bid(state):
    """AI model predicts the bid for a robot player."""
    logits, _ = robot_forward_pass.apply(robot_params, state.observation)

    # Mask illegal actions
    masked_logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
    pi = distrax.Categorical(logits=masked_logits)
    action = int(pi.mode().item())  # Select highest probability action

    return action, BID_LABELS[action]


# Initialize game state
rng = jax.random.PRNGKey(0)
state = eval_env.init(rng)

print("Bridge Bidding Game Started!")
print(
    f"Human players: {[seat for seat, is_human in player_config.items() if is_human]}"
)

while not state.terminated.all():  # Run bidding until the phase ends
    current_player = int(state.current_player.item())  # Convert JAX array to integer
    seat_name = player_seat_map[current_player]
    is_human = player_config[seat_name]

    parse_state(state, current_player, seat_name)

    if is_human:
        # Display available legal actions
        legal_actions = jnp.where(state.legal_action_mask)[0]  # Get legal bids
        readable_legal_actions = [BID_LABELS[i] for i in legal_actions]

        # Prompt human player
        print(f"{seat_name}, what is your bid? Legal bids: {readable_legal_actions}")

        # Get user input bid
        while True:
            try:
                user_bid_str = input("Enter your bid: ").strip().upper()
                if user_bid_str in readable_legal_actions:
                    user_bid = BID_LABELS.index(
                        user_bid_str
                    )  # Convert bid string to index
                    break
                else:
                    print("Invalid bid! Choose from:", readable_legal_actions)
            except ValueError:
                print("Please enter a valid bid.")

        action = user_bid
    else:
        # Robot makes a bid
        action, action_str = predict_bid(state)
        print(f"{seat_name} (Robot) bids: {action_str}")

    # Apply the bid to the game state
    state = eval_env.step(state, action, rng)

print("Bidding phase ended.")
