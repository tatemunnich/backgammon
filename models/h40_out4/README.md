# 40 hidden units, 4 outputs — near-best weights

`ckpt-39.npz` is the strongest network found so far at this size (198 inputs, 40 hidden
units, 4 outputs: black win, black gammon, white win, white gammon). Longer training and
changes to the learning rates and lambda no longer improve it beyond measurement noise, so
treat it as close to the ceiling for this network size.

Load it with:

```python
from players.NeuralNetPlayer import NeuralNet, NeuralNetPlayer
net = NeuralNet(checkpoint_dir="models/h40_out4")
net.load(num=39)
player = NeuralNetPlayer(color, net)
```

## Where it came from

Copied from `checkpoints/td_lam03/ckpt-39.npz`
(sha256 `05b214c1fabaef6079370f5cbd1f1c70e744fd7012f9fbac0927b631680e1528`).
`history.jsonl` is that run's full training log.

| Games | Alpha | Beta | Lambda | Notes |
|---|---|---|---|---|
| 0 – 90k | 1/198 | 1/40 | 0.6 | default rates (`td_4out`, ckpt-1..18) |
| 90k – 105k | not recorded | | | ckpt-19..21 |
| 105k – 135k | 0.02 | 0.1 | 0.6 | ckpt-22..27 |
| 135k – 165k | 1/198 | 1/40 | 0.6 | ckpt-28..33 (`td_lr_quarter`) |
| 165k – 195k | 0.0025 | 0.0125 | 0.3 | ckpt-34..39 (`td_lam03`, commit 782c0fc) |

## How strong it is

`evaluate.py`, 2,000 games per match (1,000 mirrored-dice pairs), seed 1:

| Opponent | Win % | Points/game | ±95% |
|---|---|---|---|
| ckpt-33 (165k) | 52.65 | +0.075 | 0.071 |
| ckpt-27 (135k) | 54.00 | +0.128 | 0.078 |

Runs compared from ckpt-33 over the last 30k games, as the mean points per game of
ckpt-34..39 against ckpt-33: lambda 0.3 +0.044, lambda 0 +0.033, lambda 0.6 +0.027
(alpha 0.0025, beta 0.0125 in all three). The differences are within noise; this checkpoint
had the best single result.
