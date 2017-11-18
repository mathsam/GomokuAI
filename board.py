import numpy as np

EMPTY = 0
PLAYER_A = 1
PLAYER_B = 2

TIE = 3


class Board(object):

    def __init__(self, num_rows, num_cols, current_player=PLAYER_A, initial_state=None):
        """
        Initialized a 2d board with shape (num_rows, num_cols)
        :param num_rows: number of rows
        :param num_cols: number of columns
        :param current_player: first player to start the game
        :param initial_state: _board may not necessarily start empty but from a previous state. Depending on the type,
            _board can be initialized in the following ways:
            str: state serialized by the method _serialize
            list|tuple: list of positions in the format [(player, row, col), ...]
        """
        self._board = [EMPTY] * (num_rows * num_cols)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._current_player = current_player
        if isinstance(initial_state, str):
            self._current_player, self._board = self._deserialize(initial_state)
        elif isinstance(initial_state, (list, tuple)):
            for p, i, j in initial_state:
                self[i,j] = p

    def update_state(self, new_state):
        self._current_player, self._board = self._deserialize(new_state)

    def convert_into_2d_array(self):
        """Convert the board into 2d numpy array"""
        return np.array([self._board[i*self.num_cols:(i+1)*self.num_cols] for i in range(self.num_rows)],
                        dtype=np.int8)

    def __getitem__(self, item):
        i, j = item
        idx = i*self.num_cols + j
        return self._board[idx]

    def __setitem__(self, key, value):
        i, j = key
        idx = i*self.num_cols + j
        self._board[idx] = value

    def _serialize(self):
        """Return a string that contains the state information including current player and _board"""
        return str((self._current_player, self._board))

    def _serialize_potential_move(self, potential_move, potential_player):
        i, j = potential_move
        origs = self[i, j], self._current_player
        self[i, j] = self._current_player = potential_player
        str_repr = self._serialize()
        self[i, j], self._current_player = origs
        return str_repr

    def _deserialize(self, state):
        return eval(state)

    def current_player(self, state=None):
        """Return current player stored in this Board instance or decoded from input `state`
        :param state: None|str
        """
        if state:
            if state[1] == str(PLAYER_A):
                return PLAYER_A
            elif state[1] == str(PLAYER_B):
                return PLAYER_B
            else:
                raise ValueError('Cannot decode input state')
        return self._current_player

    def __str__(self):
        s = ''
        for i in range(self.num_rows):
            s += ''.join([str(x) for x in self._board[i*self.num_cols:(i+1)*self.num_cols]]) + '\n'
        return s

    def __repr__(self):
        return self.__str__()
