import unittest

from board.Board import Board, BLACK, WHITE, NONE


def make_board(points, black_bar=0, white_bar=0):
    """Build a board from {point: count}; positive = black, negative = white.
    Index 0 is black's borne-off pile, 25 is white's."""
    content = [0] * 26
    for point, count in points.items():
        content[point] = count
    black = {p for p in range(1, 25) if content[p] > 0}
    white = {p for p in range(1, 25) if content[p] < 0}
    return Board(pointsContent=content, blackCheckers=black, whiteCheckers=white,
                 blackCheckersTaken=black_bar, whiteCheckersTaken=white_bar)


class GameValueTest(unittest.TestCase):

    def test_no_winner(self):
        self.assertEqual(Board().getWinner(game_value=True), (NONE, 1))
        self.assertEqual(Board().getWinner(), NONE)

    def test_single_when_loser_has_borne_off(self):
        b = make_board({0: 15, 25: -14, 20: -1})
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 1))

    def test_single_even_if_loser_has_checker_outside_home(self):
        # White bore off 5, then had a checker hit; it is now back on point 10.
        b = make_board({0: 15, 25: -5, 10: -1, 20: -9})
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 1))

    def test_gammon_when_loser_all_home_but_none_off(self):
        b = make_board({0: 15, 19: -8, 20: -7})
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 2))

    def test_gammon_with_checkers_outside_home(self):
        b = make_board({0: 15, 10: -3, 20: -12})
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 2))

    def test_backgammon_checker_in_winners_home(self):
        # Black's home board is points 1-6.
        b = make_board({0: 15, 3: -1, 20: -14})
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 3))

    def test_backgammon_checker_on_bar(self):
        b = make_board({0: 15, 20: -14}, white_bar=1)
        self.assertEqual(b.getWinner(game_value=True), (BLACK, 3))

    def test_white_wins_mirror_cases(self):
        # White's home board is points 19-24.
        self.assertEqual(make_board({25: -15, 0: 2, 5: 13}).getWinner(game_value=True), (WHITE, 1))
        self.assertEqual(make_board({25: -15, 3: 15}).getWinner(game_value=True), (WHITE, 2))
        self.assertEqual(make_board({25: -15, 22: 1, 5: 14}).getWinner(game_value=True), (WHITE, 3))
        self.assertEqual(make_board({25: -15, 5: 14}, black_bar=1).getWinner(game_value=True), (WHITE, 3))


if __name__ == '__main__':
    unittest.main()
