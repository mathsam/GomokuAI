import itertools

import matplotlib.pyplot as plt
import numpy as np

EMPTY = 0
PLAYER_A = 1  # black stone, first player to move
PLAYER_B = 2

TIE = 3


class Board(object):

    num_rows = 11
    num_cols = 11

    def __init__(self):
        """
        Initialized a 2d board with shape (num_rows, num_cols)
        :param initial_state: None|list, _board may not necessarily start empty but from a previous state.
        """
        self._board = [EMPTY] * (Board.num_rows * Board.num_cols)
        self._empty_indices = list(itertools.product(range(Board.num_rows), range(Board.num_cols)))
        self.num_stones = 0
        self.last_player = None
        self.history = []

    def update_state(self, move):
        """Add a stone to the board. The color of the stone is automatically determined
        :param move: tuple, (row_idx, col_idx)
        :return: None
        """
        if self.num_stones % 2 == 0:
            self[move] = PLAYER_A
        else:
            self[move] = PLAYER_B
        self.last_player = self.current_player()
        self.num_stones += 1
        self._empty_indices.remove(move)
        self.history.append(move)

    def convert_into_2d_array(self):
        """Convert the board into 2d numpy array"""
        return np.array([self._board[i*self.num_cols:(i+1)*self.num_cols] for i in range(self.num_rows)],
                        dtype=np.int8)

    def current_player(self):
        """Return current player stored in this Board instance or decoded from input `state`
        """
        if self.num_stones % 2 == 0:
            return PLAYER_A
        return PLAYER_B

    def avial_moves(self):
        return self._empty_indices

    def draw(self, ax=None, pause_time=None):
        if ax is None:
            start_idx = 0
        else:
            start_idx = len(self.history) - 1  # only need to draw last stone
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xticks(range(1, Board.num_cols+1))
            ax.set_yticks(range(1, Board.num_rows+1))
            ax.grid('on')
            ax.set_xlim([0, Board.num_cols+1])
            ax.set_ylim([0, Board.num_rows+1])
            ax.set_aspect('equal')
        for t, move in enumerate(self.history[start_idx:], start_idx+1):
            y, x = move
            player = self[move]
            if player == PLAYER_A:
                ax.scatter(x+1, y+1, c='k', s=100)
            else:
                ax.scatter(x+1, y+1, c='r', s=100)
            ax.text(x+1.2, y+1, str(t), color='b')
            if pause_time:
                plt.pause(pause_time)
        return ax

    def __getitem__(self, item):
        i, j = item
        idx = i*self.num_cols + j
        return self._board[idx]

    def __setitem__(self, key, value):
        i, j = key
        idx = i*self.num_cols + j
        self._board[idx] = value

    def __str__(self):
        s = ''
        for i in range(self.num_rows):
            s += ''.join([str(x) for x in self._board[i*self.num_cols:(i+1)*self.num_cols]]) + '\n'
        return s

    def __repr__(self):
        return self.__str__()
