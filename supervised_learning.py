"""This code is modified from OpenSpiel:

  https://github.com/google-deepmind/open_spiel

Please refer to their work if you use this example in your research."""
import os
import pickle
from typing import Any

from absl import app
from absl import flags

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax

import pyspiel
import wandb
from src.models import ActorCritic, ActorNN
import distrax

OptState = Any
Params = Any

FLAGS = flags.FLAGS
GAME = pyspiel.load_game("bridge(use_double_dummy_result=false)")
NUM_ACTIONS = 38
MIN_ACTION = 52
NUM_CARDS = 52
NUM_PLAYERS = 4
TOP_K_ACTIONS = 5  # How many alternative actions to display

flags.DEFINE_integer("iterations", 400000, "Number of iterations")
flags.DEFINE_string("data_path", None, "Location for data")
flags.DEFINE_integer("eval_every", 10000, "How often to evaluate the policy")
flags.DEFINE_integer("num_examples", 3, "How many examples to print per evaluation")
flags.DEFINE_integer("train_batch", 128, "Batch size for training step")
flags.DEFINE_integer("eval_batch", 10000, "Batch size when evaluating")
flags.DEFINE_integer("rng_seed", 42, "Seed for initial network weights")
flags.DEFINE_string("save_path", None, "Location for saved networks")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for optimizer")
flags.DEFINE_float("entropy_coef", 0, "coef for entropy term")
flags.DEFINE_string("model_type", "DeepMind", "model type of NN, DeepMind or FAIR")
flags.DEFINE_string("activation", "relu", "activation for NN")


def _no_play_trajectory(line: str):
    """Returns the deal and bidding actions only given a text trajectory."""
    actions = [int(x) for x in line.split(" ")]
    # Usually a trajectory is NUM_CARDS chance events for the deal, plus one
    # action for every bid of the auction, plus NUM_CARDS actions for the play
    # phase. Exceptionally, if all NUM_PLAYERS players Pass, there is no play
    # phase and the trajectory is just of length NUM_CARDS + NUM_PLAYERS.
    if len(actions) == NUM_CARDS + NUM_PLAYERS:
        return tuple(actions)
    else:
        return tuple(actions[:-NUM_CARDS])


def make_dataset(file: str):
    """Creates dataset as a generator of single examples."""
    all_trajectories = [_no_play_trajectory(line) for line in open(file)]
    while True:
        np.random.shuffle(all_trajectories)
        for trajectory in all_trajectories:
            action_index = np.random.randint(52, len(trajectory))
            state = GAME.new_initial_state()
            for action in trajectory[:action_index]:
                state.apply_action(action)
            legal_actions = state.legal_actions()
            legal_actions_tensor = np.zeros(38)
            legal_actions_tensor[np.array(legal_actions) - 52] = 1
            yield (
                state.observation_tensor()[4:484],
                trajectory[action_index] - MIN_ACTION,
                legal_actions_tensor,
            )


def batch(dataset, batch_size: int):
    """Creates a batched dataset from a one-at-a-time dataset."""
    observations = np.zeros([batch_size] + [480], np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)
    legal_actions = np.zeros([batch_size] + [38], dtype=np.bool_)
    while True:
        for batch_index in range(batch_size):
            (
                observations[batch_index],
                labels[batch_index],
                legal_actions[batch_index],
            ) = next(dataset)
        yield observations, labels, legal_actions


def one_hot(x, k):
    """Returns a one-hot encoding of `x` of size `k`."""
    return jnp.array(x[..., jnp.newaxis] == jnp.arange(k), dtype=np.float32)


def actor_critic_net_fn(x):
    net = ActorCritic(
        action_dim=38, activation=FLAGS.activation, model=FLAGS.model_type
    )
    logits, value = net(x)
    return logits


def main(argv):
    wandb.init(
        project="wbride5_learning",
        config={
            "ITERATIONS": FLAGS.iterations,
            "TRAIN_BATCH": FLAGS.train_batch,
            "lr": FLAGS.learning_rate,
            "model_type": FLAGS.model_type,
            "activation": FLAGS.activation,
            "entropy_coef": FLAGS.entropy_coef,
        },
    )
    config = wandb.config
    print(config)
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Make the network.
    net = hk.without_apply_rng(hk.transform(actor_critic_net_fn))
    # Make the optimiser.
    opt = optax.adam(FLAGS.learning_rate)

    @jax.jit
    def loss(
        params: Params,
        inputs: np.ndarray,
        targets: np.ndarray,
        legal_actions: np.ndarray,
    ):
        """Cross-entropy loss."""
        assert targets.dtype == np.int32
        logits = net.apply(params, inputs)
        log_probs = jax.nn.log_softmax(logits)
        target_loss = -jnp.mean(one_hot(targets, NUM_ACTIONS) * log_probs)
        masked_logits = logits + jnp.finfo(np.float64).min * (~legal_actions)
        masked_pi = distrax.Categorical(masked_logits)
        entropy = masked_pi.entropy().mean()
        total_loss = target_loss - FLAGS.entropy_coef * entropy
        return total_loss, (target_loss, entropy)

    @jax.jit
    def accuracy(
        params: Params,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> jax.Array:
        """Classification accuracy."""
        predictions = jax.nn.softmax(net.apply(params, inputs))
        return jnp.mean(jnp.argmax(predictions, axis=-1) == targets)

    @jax.jit
    def update(
        params: Params,
        opt_state: OptState,
        inputs: np.ndarray,
        targets: np.ndarray,
        legal_actions: np.ndarray,
    ):
        """Learning rule (stochastic gradient descent)."""
        grad_fn = jax.value_and_grad(loss, has_aux=True)
        Loss, gradient = grad_fn(params, inputs, targets, legal_actions)
        updates, opt_state = opt.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, Loss

    def output_samples(params: Params, max_samples: int):
        """Output some cases where the policy disagrees with the dataset action."""
        if max_samples == 0:
            return
        count = 0
        with open(os.path.join(FLAGS.data_path, "test.txt")) as f:
            lines = list(f)
        np.random.shuffle(lines)
        for line in lines:
            state = GAME.new_initial_state()
            actions = _no_play_trajectory(line)
            for action in actions:
                if not state.is_chance_node():
                    observation = np.array(
                        state.observation_tensor()[4:484], np.float32
                    )
                    # policy = np.exp(net.apply(params, observation))
                    policy = jax.nn.softmax(net.apply(params, observation))
                    probs_actions = [(p, a + MIN_ACTION) for a, p in enumerate(policy)]
                    pred = max(probs_actions)[1]
                    if pred != action:
                        print(state)
                        for p, a in reversed(sorted(probs_actions)[-TOP_K_ACTIONS:]):
                            print("{:7} {:.2f}".format(state.action_to_string(a), p))
                        print(
                            "Ground truth {}\n".format(state.action_to_string(action))
                        )
                        count += 1
                        break
                state.apply_action(action)
            if count >= max_samples:
                return

    # Make datasets.
    if FLAGS.data_path is None:
        raise app.UsageError(
            "Please generate your own supervised training data or download from "
            "https://console.cloud.google.com/storage/browser/openspiel-data/bridge"
            " and supply the local location as --data_path"
        )
    train = batch(
        make_dataset(os.path.join(FLAGS.data_path, "train.txt")),
        FLAGS.train_batch,
    )
    test = batch(
        make_dataset(os.path.join(FLAGS.data_path, "test.txt")),
        FLAGS.eval_batch,
    )

    # Initialize network and optimiser.
    rng = jax.random.PRNGKey(FLAGS.rng_seed)  # seed used for network weights
    inputs, unused_targets, unused_legal_actions = next(train)
    params = net.init(rng, inputs)
    opt_state = opt.init(params)

    # Train/eval loop.
    for step in range(FLAGS.iterations):
        # Do SGD on a batch of training examples.
        inputs, targets, legal_actions = next(train)
        params, opt_state, train_loss = update(
            params, opt_state, inputs, targets, legal_actions
        )
        total_loss, (target_loss, entropy) = train_loss
        train_accuracy = accuracy(params, inputs, targets)
        metrics = {
            "train/total_loss": total_loss,
            "train/target_loss": target_loss,
            "train/entropy": entropy,
            "train/train_accuracy": train_accuracy,
        }

        # Periodically evaluate classification accuracy on the test set.
        if (1 + step) % FLAGS.eval_every == 0:
            inputs, targets, legal_actions = next(test)
            test_accuracy = accuracy(params, inputs, targets)
            test_loss = loss(params, inputs, targets, legal_actions)
            total_loss, (target_loss, entropy) = test_loss
            pi = jax.nn.softmax(net.apply(params, inputs))
            illegal_action_prob = jax.vmap(jnp.dot)(pi, ~legal_actions)
            print(f"After {1+step} steps, test accuracy: {test_accuracy}.")
            test_metrics = {
                "test/total_loss": total_loss,
                "test/target_loss": target_loss,
                "test/entropy": entropy,
                "test/test_accuracy": test_accuracy,
                "test/illegal_actions_prob": illegal_action_prob.mean(),
            }
            if FLAGS.save_path is not None:
                filename = os.path.join(FLAGS.save_path, f"params-{1 + step}.pkl")
                with open(filename, "wb") as pkl_file:
                    pickle.dump(params, pkl_file)
            output_samples(params, FLAGS.num_examples)
            wandb.log({**test_metrics})
        wandb.log({**metrics})
    wandb.finish()


if __name__ == "__main__":
    app.run(main)