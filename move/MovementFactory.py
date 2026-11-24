"""Legal-play generation.

generate_moves() returns one MoveNode per distinct position the roll can lead to.

How it works: the board is converted once into the mover's point of view, as two plain lists indexed by
the mover's point numbers (24 = farthest from home, 1 = deepest in home):

    me[r]   the mover's checkers on point r; me[0] = borne off, me[25] = on the bar
    opp[r]  the opponent's checkers on that same point; opp[0] counts opponent checkers hit during the play

so every move is "from r to r - die", whichever color is moving. The search then plays out each die
sequence, copying the two small lists at each step, and only builds Board objects for the plays that
survive at the end (their move names are built only if something prints them).

Rules enforced: a player must use as many dice as possible, and if only one die can be used, the larger
one when possible. Checkers on the bar must enter before anything else moves. Bearing off needs every
checker in the home board, and a die larger than needed may only bear off from the farthest-back point.
"""
from board.Board import Board, BLACK
from board.Dice import Dice
from move.Move import MoveNode


def generate_moves(board: Board, color: str, dice: Dice, verbose=False):
    pts = board.pointsContent
    is_black = color == BLACK
    if is_black:  # black's point numbers already count down to home
        me = [p if p > 0 else 0 for p in pts]
        opp = [-p if p < 0 else 0 for p in pts]
        me[25] = board.blackCheckersTaken
    else:  # white's point r is absolute point 25 - r
        rev = pts[::-1]
        me = [-p if p < 0 else 0 for p in rev]
        opp = [p if p > 0 else 0 for p in rev]
        me[25] = board.whiteCheckersTaken
    opp[0] = opp[25] = 0  # opponent's off/bar aren't needed; opp[0] becomes the hit count

    die_1, die_2 = dice.getDice()
    high = max(die_1, die_2)
    doubles = die_1 == die_2
    sequences = [(die_1,) * 4] if doubles else [(die_1, die_2), (die_2, die_1)]

    # Finished plays: position key -> [me, opp, path, used the larger die]. Only plays using the most
    # dice are kept. path is a tuple of (from, to, hit) steps in the mover's point numbers.
    results = {}
    best = 0

    def record(me, opp, used, path, big):
        nonlocal best
        if used < best:
            return
        if used > best:
            results.clear()
            best = used
        key = (tuple(me), opp[0] and tuple(opp))  # opp only differs between plays if something was hit
        entry = results.get(key)
        if entry is None:
            results[key] = [me, opp, path, big]
        elif big and not entry[3]:
            entry[2], entry[3] = path, True

    def search(me, opp, seq, i, start, path):
        """Play die seq[i] from every legal point at or below `start`, then recurse.

        `start` is the previous move's starting point, so checkers are moved back to front. Every
        play is still reached, because any legal sequence of moves can be rearranged that way
        (moving my own checkers never blocks anything, and a back checker that could bear off later
        can also bear off first), while far fewer duplicate orderings are tried. Different dice
        still try both dice orders, since which die goes first matters.
        """
        if i < len(seq):
            n = seq[i]
            moved = False
            if me[25]:  # on the bar: must enter first
                t = 25 - n
                if opp[t] < 2:
                    me2 = me[:]
                    me2[25] -= 1
                    me2[t] += 1
                    hit = opp[t] == 1
                    if hit:
                        opp2 = opp[:]
                        opp2[t] = 0
                        opp2[0] += 1
                    else:
                        opp2 = opp
                    search(me2, opp2, seq, i + 1, 25, path + ((25, t, hit),))
                    moved = True
            else:
                far = 24
                while far and not me[far]:
                    far -= 1
                for r in range(min(start, far), 0, -1):
                    if not me[r]:
                        continue
                    t = r - n
                    if t > 0:
                        c = opp[t]
                        if c >= 2:
                            continue
                        me2 = me[:]
                        me2[r] -= 1
                        me2[t] += 1
                        if c:
                            opp2 = opp[:]
                            opp2[t] = 0
                            opp2[0] += 1
                        else:
                            opp2 = opp
                        search(me2, opp2, seq, i + 1, r, path + ((r, t, c == 1),))
                    elif far <= 6 and (t == 0 or r == far):  # bear off
                        me2 = me[:]
                        me2[r] -= 1
                        me2[0] += 1
                        search(me2, opp, seq, i + 1, r, path + ((r, 0, False),))
                    else:
                        continue
                    moved = True
            if moved:
                return
        record(me, opp, i, path, i == 1 and seq[0] == high)

    for seq in sequences:
        search(me, opp, seq, 0, 25, ())

    plays = list(results.values())
    if best == 1 and not doubles and any(big for _, _, _, big in plays):
        plays = [p for p in plays if p[3]]  # only one die can be used: it must be the larger one

    prefix = color + " " + str(dice)
    moves = []
    for me, opp, path, _ in plays:
        if is_black:
            new_pts = [a - b for a, b in zip(me, opp)]
            new_pts[0] = me[0]
            new_pts[25] = pts[25]
            black_bar, white_bar = me[25], board.whiteCheckersTaken + opp[0]
        else:
            new_pts = [b - a for a, b in zip(me, opp)]
            new_pts.reverse()
            new_pts[0] = pts[0]
            new_pts[25] = -me[0]
            black_bar, white_bar = board.blackCheckersTaken + opp[0], me[25]
        after = Board(new_pts, None, None, black_bar, white_bar, board.doubleCube, board.doublePossession)
        moves.append(Play(prefix, path, is_black, after, best))

    if verbose:
        print(board)
        print(dice)
        print(moves)

    return moves


class Play(MoveNode):
    """A generated play. Its name is only built if something asks for it (printing, HumanPlayer)."""

    def __init__(self, prefix, path, is_black, board_after, deep):
        self.children = []
        self.board_after = board_after
        self.deep = deep
        self.die = None
        self._name = None
        self._parts = (prefix, path, is_black)

    @property
    def name(self):
        if self._name is None:
            self._name = _play_name(*self._parts)
        return self._name


def _play_name(prefix, path, is_black):
    """e.g. "BLACK 6-4: 24/18 13/9*", using absolute point numbers."""
    parts = [prefix]
    for r, t, hit in path:
        if not is_black:
            r, t = 25 - r, 25 - t
        if r in (0, 25):
            parts.append("bar/" + str(t) + ("*" if hit else ""))
        elif t in (0, 25):
            parts.append(str(r) + "/off")
        else:
            parts.append(str(r) + "/" + str(t) + ("*" if hit else ""))
    return " ".join(parts)
