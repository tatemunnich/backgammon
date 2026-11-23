import json
import os
import re

import numpy as np

from board.Board import getPieceSymbol, BLACK, Board, WHITE, getOtherColor, NONE
from move.MovementFactory import generate_moves
from players.Player import Player


class NeuralNetPlayer(Player):
    def __init__(self, color, network, name="Net", learning=False):
        self.name = name
        self.network = network
        self.color = color
        self.learning = learning

    # def get_move(self, backgammon, return_value=False):
    #     moves = generate_moves(backgammon.board, BLACK, backgammon.dice)
    #     best, value = None, 0
    #     for move in moves:
    #         new_value = self.network.calculate
    #         if new_value > value:
    #             best = move
    #             value = new_value
    #     if return_value:
    #         return best, value
    #     else:
    #         return best

    def get_move(self, backgammon):
        # pa4 specs
        to_move = self.color
        next_player = getOtherColor(self.color)

        board = backgammon.board
        moves = list(generate_moves(board, to_move, backgammon.dice))

        # Expected network output after each candidate move, one row per move.
        outputs = np.empty((len(moves), self.network.num_outputs))
        unfinished = []
        for i, move in enumerate(moves):
            winner, game_value = move.board_after.getWinner(game_value=True)
            if winner != NONE:
                # The game ends here: score the move by its real result, not the network's guess.
                outputs[i] = self.outcome_outputs(winner, game_value)[0]
            else:
                unfinished.append(i)
        if unfinished:
            # Score every other candidate in one batch: one input matrix, one pass through the network.
            inputs = self.get_input_vectors([moves[i].board_after for i in unfinished], next_player)
            outputs[unfinished] = self.network.predict(inputs)

        best = int(np.argmax(self.value_for(outputs, to_move)))  # first best move, as before

        if self.learning:
            # TD step: move V(current position) toward V(chosen next position). When the chosen
            # move wins, the target is the true outcome, so this is also the final update of the game.
            self.evaluate(board, to_move, to_move)
            self.network.backprop(outputs[best:best + 1])

        return moves[best]

    def new_game(self):
        if self.learning:
            self.network.reset_traces()

    def outcome_outputs(self, winner, game_value):
        """The network output a finished game should have produced."""
        if self.network.num_outputs == 1:  # P(black wins)
            return np.array([[1.]]) if winner == BLACK else np.array([[0.]])
        elif self.network.num_outputs == 4:  # black win, black gammon, white win, white gammon
            gammon = game_value > 1
            if winner == BLACK:
                return np.array([[0., 1., 0., 0.]]) if gammon else np.array([[1., 0., 0., 0.]])
            else:
                return np.array([[0., 0., 0., 1.]]) if gammon else np.array([[0., 0., 1., 0.]])
        raise Exception("Could not evaluate output shape")

    def __str__(self):
        return self.name + " (" + getPieceSymbol(self.color) + ")"

    @staticmethod
    def get_input_vectors(boards, current_color=BLACK):
        """Encode several boards at once, one row each: shape (len(boards), 198).

             BLACK_OFF | BLACK_POINTS_CONTENT | BLACK_BAR | TURN | WHITE_BAR | WHITE_POINTS CONTENT | WHITE_OFF
                 1            24 * 4 = 96          1         2         1           24 * 4 = 96            1

        Each point uses 4 values for its checker count c: c>=1, c>=2, c>=3, and (c-3)/2 when c>3.
        """
        n = len(boards)
        content = np.array([b.pointsContent for b in boards], dtype=float)  # (n, 26)
        points = content[:, 1:25]

        def encode(counts):  # (n, 24) -> (n, 96)
            units = np.stack([counts >= 1, counts >= 2, counts >= 3, np.maximum(counts - 3, 0) / 2], axis=2)
            return units.reshape(n, 96)

        bars = np.array([(b.numBar(BLACK), b.numBar(WHITE)) for b in boards], dtype=float) / 2
        turn = np.tile([1., 0.] if current_color == BLACK else [0., 1.], (n, 1))
        return np.hstack([np.abs(content[:, :1]) / 15,         # black off
                          encode(np.maximum(points, 0)),       # black checkers
                          bars[:, :1],                         # black bar
                          turn,
                          bars[:, 1:],                         # white bar
                          encode(np.maximum(-points, 0)),      # white checkers
                          np.abs(content[:, 25:]) / 15])       # white off

    @staticmethod
    def get_input_vector(board=None, current_color=BLACK):
        """Encode one board: shape (1, 198). See get_input_vectors."""
        return NeuralNetPlayer.get_input_vectors([board if board is not None else Board()], current_color)

    def evaluate(self, board, color_to_move, color):
        x = self.get_input_vector(board, color_to_move)
        outputs = self.network.calculate(x)
        return self.value_for(outputs, color)[0], outputs

    def value_for(self, outputs, color):
        """Equity of each row of `outputs` (shape (n, num_outputs)) from `color`'s point of view."""
        outputs = np.asarray(outputs)
        if self.network.num_outputs == 1:  # P(black wins)
            black = outputs[:, 0]
            return black if color == BLACK else 1 - black
        elif self.network.num_outputs == 4:  # black win, black gammon, white win, white gammon
            black = outputs @ np.array([1., 2., -1., -2.])
            return black if color == BLACK else -black
        raise Exception("Could not evaluate output shape")


class NeuralNet:
    """A 198-40-N sigmoid network trained with TD(lambda), written with NumPy.

    Every array keeps a leading batch axis of 1: inputs (1, num_inputs), hidden outputs
    (1, num_hidden), outputs (1, num_outputs).
    """

    # Starting weights are drawn from U(-INIT_RANGE, INIT_RANGE). About 27 inputs are active
    # in a typical position, so +-0.1 keeps each hidden unit's input near 0, where the sigmoid
    # is steepest and learns fastest, and starts the output near 0.5 (no opinion). The old
    # +-1 range started units around +-3, already close to saturated.
    INIT_RANGE = 0.1

    # Default TD(lambda) settings, as specified in Gettysburg PA4.
    DEFAULT_LAMBDA = 0.6

    def __init__(self, checkpoint_dir, num_inputs=198, num_hidden=40, num_outputs=1,
                 hidden_weights=None, second_weights=None, save_traces=False,
                 alpha=None, beta=None, lambd=None):
        """alpha: hidden-layer learning rate (default 1/num_inputs).
        beta: output-layer learning rate (default 1/num_hidden).
        lambd: eligibility trace decay per step (default DEFAULT_LAMBDA).
        Any left as None are taken from the checkpoint by load(), when it recorded them."""
        self.inputs = None  # shape (1, num_inputs)
        self.hidden_outputs = None  # shape (1, num_hidden)
        self.outputs = None  # shape (1, num_outputs)

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        if hidden_weights is not None:
            self.hidden_weights = np.array(hidden_weights, dtype=float)  # shape (num_inputs, num_hidden)
            self.second_weights = np.array(second_weights, dtype=float)  # shape (num_hidden, num_outputs)
        else:
            self.randomize_weights()

        self.checkpoint_dir = checkpoint_dir
        self.save_traces = save_traces
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.games_trained = None  # set by load() when the checkpoint recorded it
        self.saved_rates = None  # (alpha, beta, lambda) recorded in the loaded checkpoint, if any
        self.reset_traces()

    @property
    def learning_rates(self):
        """(alpha, beta, lambda) actually used by backprop()."""
        alpha = self.alpha if self.alpha is not None else 1 / self.num_inputs
        beta = self.beta if self.beta is not None else 1 / self.num_hidden
        lambd = self.lambd if self.lambd is not None else self.DEFAULT_LAMBDA
        return alpha, beta, lambd

    def reset_traces(self):
        """Zero the eligibility traces. Call at the start of every training game."""
        self.ew = np.zeros((self.num_hidden, self.num_outputs))
        self.ev = np.zeros((self.num_inputs, self.num_hidden, self.num_outputs))

    def randomize_weights(self):
        r = self.INIT_RANGE
        self.hidden_weights = np.random.uniform(-r, r, size=(self.num_inputs, self.num_hidden))
        self.second_weights = np.random.uniform(-r, r, size=(self.num_hidden, self.num_outputs))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def calculate(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = self.sigmoid(inputs @ self.hidden_weights)
        self.outputs = self.sigmoid(self.hidden_outputs @ self.second_weights)
        return self.outputs

    def predict(self, inputs):
        """Outputs for a batch of inputs (n, num_inputs) -> (n, num_outputs).

        Unlike calculate(), this leaves the state that backprop() uses untouched.
        """
        return self.sigmoid(self.sigmoid(inputs @ self.hidden_weights) @ self.second_weights)

    # ---- Saving and loading -------------------------------------------------------------
    # save() writes numbered checkpoints, <checkpoint_dir>/ckpt-1.npz, ckpt-2.npz, ...
    # save_to_text() writes the weights and traces as JSON into checkpoint_dir.

    def checkpoint_numbers(self):
        numbers = []
        if os.path.isdir(self.checkpoint_dir):
            for name in os.listdir(self.checkpoint_dir):
                match = re.fullmatch(r"ckpt-(\d+)\.npz", name)
                if match:
                    numbers.append(int(match.group(1)))
        return sorted(numbers)

    def save(self, games=None):
        """Write the next numbered checkpoint. `games` (games trained so far) is stored with it,
        and so are the learning rates the weights were trained with."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        numbers = self.checkpoint_numbers()
        num = numbers[-1] + 1 if numbers else 1
        arrays = {"hidden": self.hidden_weights, "second": self.second_weights}
        if games is not None:
            arrays["games"] = np.array(games)
        arrays.update(zip(("alpha", "beta", "lambda"), map(np.array, self.learning_rates)))
        if self.save_traces:
            arrays.update(ew=self.ew, ev=self.ev)
        path = os.path.join(self.checkpoint_dir, "ckpt-%d.npz" % num)
        # Write to a temporary file and rename, so a reader never sees a half-written checkpoint.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
        return path

    def save_to_text(self, filename):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.checkpoint_dir, filename), 'w') as f:
            weights = {"hidden": self.hidden_weights.tolist(), "second": self.second_weights.tolist(),
                       "ew": self.ew.tolist(), "ev": self.ev.tolist()}
            json.dump(weights, f)

    def load(self, num=None, text_path=None):
        """Load weights from a JSON file (text_path), checkpoint number `num`, or the latest checkpoint."""
        if text_path:
            with open(text_path) as f:
                saved = json.load(f)
        else:
            if num is None:
                numbers = self.checkpoint_numbers()
                if not numbers:
                    raise FileNotFoundError("No checkpoints (ckpt-N.npz) in " + self.checkpoint_dir)
                num = numbers[-1]
            with np.load(os.path.join(self.checkpoint_dir, "ckpt-%d.npz" % num)) as data:
                saved = {key: data[key] for key in data.files}

        self.hidden_weights = np.array(saved["hidden"], dtype=float)
        self.second_weights = np.array(saved["second"], dtype=float)
        self.num_inputs, self.num_hidden = self.hidden_weights.shape
        self.num_outputs = self.second_weights.shape[1]
        self.games_trained = int(saved["games"]) if "games" in saved else None
        # Learning rates the checkpoint was trained with (None for older checkpoints). Rates not
        # set explicitly continue from the checkpoint, so --resume keeps the run's rates.
        self.saved_rates = tuple(float(saved[k]) for k in ("alpha", "beta", "lambda")) \
            if all(k in saved for k in ("alpha", "beta", "lambda")) else None
        if self.saved_rates:
            if self.alpha is None:
                self.alpha = self.saved_rates[0]
            if self.beta is None:
                self.beta = self.saved_rates[1]
            if self.lambd is None:
                self.lambd = self.saved_rates[2]
        if "ew" in saved and "ev" in saved:
            self.ew = np.array(saved["ew"], dtype=float)
            self.ev = np.array(saved["ev"], dtype=float)
        else:
            self.reset_traces()
        self.inputs = None
        self.hidden_outputs = None
        self.outputs = None

    # ---- TD(lambda) learning ------------------------------------------------------------

    @staticmethod
    def gradient(unit):
        """Slope of the sigmoid, given its output."""
        return unit * (1 - unit)

    def backprop(self, expected):
        """Move the output for the last calculate() call toward `expected`.

        As specified in Gettysburg PA4: ew and ev are eligibility traces for the output and
        hidden weights, decayed by lambda each step.
        """
        alpha, beta, lambd = self.learning_rates
        out_slope = self.gradient(self.outputs)  # (1, O)
        # d output_k / d second_weights[j, k] = h_j * y_k(1 - y_k)            -> (H, O)
        self.ew = lambd * self.ew + np.tensordot(self.hidden_outputs, out_slope, axes=[0, 0])
        # d output_k / d hidden_weights[i, j] = x_i * h_j(1 - h_j) * w_jk * y_k(1 - y_k)  -> (I, H, O)
        hidden_term = np.tensordot(self.gradient(self.hidden_outputs), out_slope, axes=[0, 0]) * self.second_weights
        self.ev = lambd * self.ev + np.tensordot(self.inputs[0], hidden_term, axes=0)

        error = expected - self.outputs  # (1, O)

        self.second_weights += beta * (self.ew * error)
        self.hidden_weights += alpha * np.tensordot(self.ev, error[0], axes=1)
