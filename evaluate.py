"""Evaluate saved checkpoints with low-noise matches, without pausing training.

Each match is played as mirrored pairs: two games with the same dice, the second with the
players' seats swapped. Much of the dice luck cancels out, so results settle with fewer games.

Examples:
    python evaluate.py                                   # latest checkpoint vs previous and Random
    python evaluate.py --checkpoints all                 # every checkpoint in the folder
    python evaluate.py --opponents previous,tate --tate-games 200
    python evaluate.py --watch                           # keep evaluating new checkpoints as training saves them

Results are printed and appended to <checkpoint-dir>/eval.csv.
"""
import argparse
import csv
import math
import os
import random
import time
from datetime import datetime

import numpy as np

from Backgammon import Backgammon
from board.Board import BLACK, WHITE
from players.NeuralNetPlayer import NeuralNetPlayer, NeuralNet
from players.RandomPlayer import RandomPlayer
from players.TatePlayer import TatePlayer

CSV_FIELDS = ["time", "checkpoint", "games_trained", "opponent", "games",
              "win_pct", "points_per_game", "ci95"]


def load_net(checkpoint_dir, num):
    net = NeuralNet(checkpoint_dir=checkpoint_dir)
    net.load(num=num)
    return net


def games_trained(checkpoint_dir, num, net):
    """Games trained when checkpoint `num` was saved: stored in the checkpoint, or else
    matched to the num-th <game>.txt snapshot (older checkpoints didn't store it)."""
    if net.games_trained is not None:
        return net.games_trained
    snapshots = sorted(int(n[:-4]) for n in os.listdir(checkpoint_dir) if n.endswith(".txt") and n[:-4].isdigit())
    return snapshots[num - 1] if num <= len(snapshots) else None


def play_pairs(make_a, make_b, pairs, seed):
    """Play `pairs` mirrored pairs between A and B. Returns A's points per game, one list per pair.

    make_a / make_b take a color and return a player of that color."""
    results = []
    for p in range(pairs):
        pair_seed = seed * 1_000_003 + p
        pair = []
        for a_is_black in (True, False):
            a = make_a(BLACK if a_is_black else WHITE)
            b = make_b(WHITE if a_is_black else BLACK)
            game = Backgammon(a, b) if a_is_black else Backgammon(b, a)
            game.set_dice_seed(pair_seed)  # same dice in both games of the pair
            random.seed(pair_seed)         # repeatable choices for players that use randomness
            winner, value = game.run()
            pair.append(value if winner == a.color else -value)
        results.append(pair)
    return results


def summarize(results):
    points = np.array(results, dtype=float)          # shape (pairs, 2)
    games = points.size
    win_pct = 100 * np.mean(points > 0)
    ppg = points.mean()
    pair_means = points.mean(axis=1)
    se = pair_means.std(ddof=1) / math.sqrt(len(pair_means)) if len(pair_means) > 1 else float("nan")
    return games, win_pct, ppg, 1.96 * se


def make_opponent(name, checkpoint_dir, num, available):
    """Return (label, color -> player) for an opponent name, or None if it doesn't apply.
    `available` lists the checkpoint numbers in the folder."""
    if name == "random":
        return "Random", lambda color: RandomPlayer(color)
    if name == "tate":
        return "Tate Jr.", lambda color: TatePlayer(color)
    if name == "previous":
        earlier = [n for n in available if n < num]  # skip gaps if a checkpoint was deleted
        if not earlier:
            return None
        other = earlier[-1]
    elif name.startswith("ckpt-"):
        other = int(name[5:])
        if other == num:
            return None
        if other not in available:
            raise SystemExit("No checkpoint ckpt-%d in %s" % (other, checkpoint_dir))
    else:
        raise ValueError("Unknown opponent " + repr(name) + ": use random, tate, previous or ckpt-N")
    net = load_net(checkpoint_dir, other)
    return "ckpt-%d" % other, lambda color: NeuralNetPlayer(color, net)


def already_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {(row["checkpoint"], row["opponent"]) for row in csv.DictReader(f)}


def append_row(csv_path, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def evaluate_checkpoint(args, num, csv_path, done, available):
    net = load_net(args.checkpoint_dir, num)
    trained = games_trained(args.checkpoint_dir, num, net)
    label = "ckpt-%d" % num + (" (%d games)" % trained if trained is not None else "")
    for name in args.opponents:
        opponent = make_opponent(name, args.checkpoint_dir, num, available)
        if opponent is None:
            continue
        opp_label, make_b = opponent
        if args.skip_done and ("ckpt-%d" % num, opp_label) in done:
            continue
        total = args.tate_games if name == "tate" else args.games
        pairs = max(1, total // 2)
        start = time.time()
        results = play_pairs(lambda color: NeuralNetPlayer(color, net), make_b, pairs, args.seed)
        games, win_pct, ppg, ci = summarize(results)
        print("%-22s vs %-9s %5d games  wins %5.1f%%  points/game %+.3f ± %.3f  (%.0fs)"
              % (label, opp_label, games, win_pct, ppg, ci, time.time() - start), flush=True)
        append_row(csv_path, {
            "time": datetime.now().isoformat(timespec="seconds"), "checkpoint": "ckpt-%d" % num,
            "games_trained": trained if trained is not None else "", "opponent": opp_label, "games": games,
            "win_pct": "%.2f" % win_pct, "points_per_game": "%.4f" % ppg, "ci95": "%.4f" % ci})
        done.add(("ckpt-%d" % num, opp_label))


def pick_checkpoints(spec, available):
    if spec == "latest":
        return available[-1:]
    if spec == "all":
        return available
    wanted = [int(x) for x in spec.split(",")]
    missing = [n for n in wanted if n not in available]
    if missing:
        raise SystemExit("No checkpoint(s) %s; available: %s" % (missing, available))
    return wanted


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved backgammon networks with mirrored-dice matches.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/td_4out")
    parser.add_argument("--checkpoints", default="latest",
                        help="latest (default), all, or a comma list of numbers like 3,5")
    parser.add_argument("--opponents", default="previous,random",
                        help="comma list of: previous, random, tate, ckpt-N (default previous,random)")
    parser.add_argument("--games", type=int, default=400, help="games per opponent (default 400 = 200 pairs)")
    parser.add_argument("--tate-games", type=int, default=100,
                        help="games against Tate Jr., which is much slower (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="dice seed; keep it fixed to compare checkpoints")
    parser.add_argument("--csv", default=None, help="results file (default <checkpoint-dir>/eval.csv)")
    parser.add_argument("--watch", action="store_true",
                        help="keep running and evaluate each new checkpoint as training saves it")
    parser.add_argument("--poll", type=int, default=30, help="seconds between checks in --watch mode")
    args = parser.parse_args()
    args.opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    args.skip_done = args.watch
    csv_path = args.csv or os.path.join(args.checkpoint_dir, "eval.csv")

    probe = NeuralNet(checkpoint_dir=args.checkpoint_dir)
    done = already_done(csv_path)
    if not args.watch:
        available = probe.checkpoint_numbers()
        if not available:
            raise SystemExit("No checkpoints (ckpt-N.npz) in " + args.checkpoint_dir)
        for num in pick_checkpoints(args.checkpoints, available):
            evaluate_checkpoint(args, num, csv_path, done, available)
        print("Results appended to", csv_path)
        return

    print("Watching", args.checkpoint_dir, "for new checkpoints (Ctrl-C to stop). Results go to", csv_path)
    evaluated = set()  # checkpoints handled in this session; finished (checkpoint, opponent) pairs are skipped via `done`
    try:
        while True:
            available = probe.checkpoint_numbers()
            for num in available:
                if num not in evaluated:
                    evaluate_checkpoint(args, num, csv_path, done, available)
                    evaluated.add(num)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
