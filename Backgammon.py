import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np

from board.Board import Board, BLACK, WHITE, NONE
from board.Dice import Dice
from players.GnuPlayer import GnuPlayer
from players.MinimaxPlayer import AlphaBetaPlayer, MinimaxPlayer
from players.NeuralNetPlayer import NeuralNetPlayer, NeuralNet
from players.RandomPlayer import RandomPlayer
from players.TatePlayer import TatePlayer
from players.heuristics import PipRatio


class Backgammon:

    def __init__(self, player1, player2):
        if not {player1.color, player2.color} == {WHITE, BLACK}:
            raise Exception("Must have one player of each color")
        self.players = [player1, player2]
        self.board = Board()
        self.rng = random  # source of dice rolls; see set_dice_seed()
        self.dice = Dice(rng=self.rng)
        self.on_roll = 0  # starting player 1

    def set_dice_seed(self, seed):
        """Roll this game's dice from their own random stream, separate from any player's randomness.

        Two games given the same seed get the same sequence of rolls, which is what lets
        evaluate.py replay a game with the players' seats swapped.
        """
        self.rng = random.Random(seed)
        self.dice.rng = self.rng

    def reset(self):
        self.board = Board()
        self.on_roll = 0

    def do_move(self, move):
        self.board.applyBoard(move.board_after)
        self.dice.roll()
        self.on_roll = (self.on_roll + 1) % 2

    def get_current_player(self):
        return self.players[self.on_roll]

    def start_game(self, verbose=False):
        player1_roll, player2_roll = 0, 0
        while player1_roll == player2_roll:
            player1_roll = self.rng.randint(1, 6)
            player2_roll = self.rng.randint(1, 6)
            if verbose:
                print(self.players[0], "rolls", str(player1_roll) + ",", self.players[1], "rolls", str(player2_roll) + ".")
        self.on_roll = 0 if player1_roll > player2_roll else 1
        self.dice.setRoll((player1_roll, player2_roll))

    def run(self, verbose=False):
        for player in self.players:
            player.new_game()
        self.start_game(verbose=verbose)
        if verbose:
            print(self.board)
            print(str(self.players[self.on_roll]) + " goes first.")
        while True:
            if verbose:
                print("")
                print(str(self.players[self.on_roll]) + " rolled " + str(self.dice))
            move = self.players[self.on_roll].get_move(self)
            self.do_move(move)
            if verbose:
                print(self.board)
                print(move)
            winner, value = self.board.getWinner(game_value=True)
            if winner != NONE:
                if self.players[0].color == winner:
                    self.players[0].won(self.board, value)
                    self.players[1].lost(self.board, value)
                else:
                    self.players[0].lost(self.board, value)
                    self.players[1].won(self.board, value)
                if verbose:
                    print("Winner, " + winner)
                return winner, value
            ##################################################
            if verbose:
                print("")
                print(str(self.players[self.on_roll]) + " rolled " + str(self.dice))
            move = self.players[self.on_roll].get_move(self)
            self.do_move(move)
            if verbose:
                print(self.board)
                print(move)
            winner, value = self.board.getWinner(game_value=True)
            if winner != NONE:
                if self.players[0].color == winner:
                    self.players[0].won(self.board, value)
                    self.players[1].lost(self.board, value)
                else:
                    self.players[0].lost(self.board, value)
                    self.players[1].won(self.board, value)
                if verbose:
                    print("Winner, " + winner)
                return winner, value

    @staticmethod
    def train(network, iters, start_trial=0, verbose=False, save_every=1000, benchmark_games=15, seed=None):
        """Self-play training. Every save_every games: save a checkpoint and a <game>.txt snapshot,
        then (if benchmark_games > 0) play that many games against Tate Jr. For larger, less noisy
        comparisons, use --benchmark-games 0 and run evaluate.py alongside.

        With a seed, game N's dice come from (seed, N), so every game gets fresh rolls, a run can be
        repeated exactly, and a resumed run carries on instead of replaying the rolls it started with."""
        p1, p2 = NeuralNetPlayer(BLACK, network, learning=True), NeuralNetPlayer(WHITE, network, learning=True)
        backgammon = Backgammon(p1, p2)
        for i in range(1, iters+1):
            if seed is not None:
                backgammon.set_dice_seed(training_dice_seed(seed, i + start_trial))
            winner, _ = backgammon.run()
            backgammon.reset()
            print("\rGame", i+start_trial, "finished", end="")
            if i == iters and i % save_every != 0:
                # Last game but not on a save boundary: save anyway so these games aren't lost.
                path = p1.network.save(games=i + start_trial)
                p1.network.save_to_text(str(i+start_trial)+".txt")
                print("\nSaved", os.path.basename(path), "after", i + start_trial, "games")
            if i % save_every == 0:
                path = p1.network.save(games=i + start_trial)
                p1.network.save_to_text(str(i+start_trial)+".txt")
                print("\nSaved", os.path.basename(path), "after", i + start_trial, "games")
                if benchmark_games > 0:
                    p1.learning = False
                    print("After", i + start_trial, "rounds of training:")
                    Backgammon.benchmark(p1, TatePlayer(WHITE), benchmark_games)
                    p1.learning = True

    @staticmethod
    def benchmark(player1, player2, num_games):
        scores = {BLACK: 0, WHITE: 0}
        wins = {BLACK: 0, WHITE: 0}
        back = Backgammon(player1, player2)
        for i in range(num_games):
            winner, value = back.run(verbose=False)
            print("\rGame", str(i+1) + ":", winner, "wins a", value, "game", end="")
            scores[winner] += value
            wins[winner] += 1
            back.reset()
        print("\nPoints:", scores)
        print("Wins:  ", wins)
        return scores


def runRandomTime():
    b = Backgammon(RandomPlayer(BLACK), GnuPlayer(WHITE))
    for i in range(3):
        print(b.run(verbose=True))
        b.reset()


def training_dice_seed(seed, game):
    """Dice seed for training game `game`. The "train:" prefix keeps these streams apart from the
    ones evaluate.py plays with, so evaluation never replays a training game's rolls."""
    return "train:%d:%d" % (seed, game)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_session(checkpoint_dir, entry):
    """Append one training session to <checkpoint_dir>/history.jsonl (one JSON object per line).

    Together the lines give the run's rate history: each session's rates apply from its
    start_game until the next session's start_game. Each checkpoint also stores its own rates."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def last_saved_game(checkpoint_dir):
    """Highest game number among the <N>.txt snapshots in checkpoint_dir, or 0 if none."""
    numbers = [int(name[:-4]) for name in os.listdir(checkpoint_dir)
               if name.endswith(".txt") and name[:-4].isdigit()] if os.path.isdir(checkpoint_dir) else []
    return max(numbers, default=0)


def main():
    parser = argparse.ArgumentParser(description="Train the TD(lambda) backgammon network by self-play.")
    parser.add_argument("--games", type=int, default=100000, help="number of self-play games to train for")
    parser.add_argument("--outputs", type=int, choices=[1, 4], default=4,
                        help="4: black win, black gammon, white win, white gammon; 1: black win only")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/td_4out",
                        help="where checkpoints (ckpt-N.npz) and <game>.txt snapshots are saved")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --checkpoint-dir")
    parser.add_argument("--start-game", type=int, default=None,
                        help="game number to count from (default: 0, or the last snapshot when resuming)")
    parser.add_argument("--save-every", type=int, default=1000, help="save (and benchmark) every N games")
    parser.add_argument("--benchmark-games", type=int, default=15, help="games against Tate Jr. after each save; 0 to skip (use evaluate.py instead)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the starting weights and the dice (default: a new random seed, "
                             "printed and saved in history.jsonl so the run can be repeated)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="hidden-layer learning rate (default 1/198, about 0.005)")
    parser.add_argument("--beta", type=float, default=None,
                        help="output-layer learning rate (default 1/40 = 0.025)")
    parser.add_argument("--lambda", dest="lambd", type=float, default=None,
                        help="eligibility trace decay, 0-1 (default %s)" % NeuralNet.DEFAULT_LAMBDA)
    args = parser.parse_args()
    # Rates left unset continue from the checkpoint when resuming (or use the defaults).

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2 ** 31)
    random.seed(args.seed)
    np.random.seed(args.seed)

    net = NeuralNet(checkpoint_dir=args.checkpoint_dir, num_outputs=args.outputs,
                    alpha=args.alpha, beta=args.beta, lambd=args.lambd)
    resumed_from = None
    if args.resume:
        resumed_from = "ckpt-%d" % net.checkpoint_numbers()[-1] if net.checkpoint_numbers() else None
        net.load()
        if args.start_game is not None:
            start = args.start_game
        elif net.games_trained is not None:  # recorded in the checkpoint
            start = net.games_trained
        else:  # older checkpoints: fall back to the newest <game>.txt snapshot
            start = last_saved_game(args.checkpoint_dir)
        print("Resuming", net.num_outputs, "output network from", args.checkpoint_dir, "at game", start)
    else:
        if net.checkpoint_numbers():
            parser.error(args.checkpoint_dir + " already has checkpoints. Use --resume to continue that run, "
                                               "or pick another --checkpoint-dir.")
        start = args.start_game or 0
        print("Starting a new", args.outputs, "output network in", args.checkpoint_dir)
    alpha, beta, lambd = net.learning_rates
    print("Learning rates: alpha %.4g, beta %.4g, lambda %.3g" % (alpha, beta, lambd))
    if net.saved_rates and net.saved_rates != (alpha, beta, lambd):
        print("  (changed from alpha %.4g, beta %.4g, lambda %.3g in %s)" % (net.saved_rates + (resumed_from,)))
    print("Seed", args.seed)
    log_session(args.checkpoint_dir, {
        "time": datetime.now().isoformat(timespec="seconds"), "start_game": start, "games": args.games,
        "alpha": alpha, "beta": beta, "lambda": lambd, "seed": args.seed, "outputs": net.num_outputs,
        "resumed_from": resumed_from, "commit": git_commit(), "command": " ".join(sys.argv)})

    Backgammon.train(net, args.games, start_trial=start, save_every=args.save_every,
                     benchmark_games=args.benchmark_games, seed=args.seed)


if __name__ == "__main__":
    main()
