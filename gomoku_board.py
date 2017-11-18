import board



class GomokuBoard(board.Board):

    search_directions = None

    def __init__(self, num_rows, num_cols, current_player=board.PLAYER_A, initial_state=None):
        board.Board.__init__(self, num_rows, num_cols, current_player, initial_state)
        if GomokuBoard.search_directions is None:
            GomokuBoard.search_directions = (
                (zip(range(num_rows), [0]*num_rows), (0, 1)),
                (zip([0]*num_cols, range(num_cols)), (1, 0)),
                (zip([0]*(num_cols-4), range(num_cols-4)), (1, 1)),
                (zip(range(num_rows-4), [0]*(num_rows-4)), (1, 1)),
                (zip([0]*(num_cols-4), range(4, num_cols)), (-1, -1)),
                (zip(range(4, num_rows), [0]*(num_rows-4)), (-1, -1))
            )
        self.search_directions = GomokuBoard.search_directions

    def next_states(self, state=None):
        pass

    def judge(self, state=None):
        """
        :param state: None(default)|str(serialized state)
        :return: Return the player that wins if exists, otherwise return None
        """
        if state:
            curr_state = GomokuBoard(self.num_rows, self.num_cols, None, state)
        else:
            curr_state = self

        if board.EMPTY not in curr_state._board:
            return board.TIE

        for start_pos, dir in self.search_directions:
            for i, j in start_pos:
                max_streak = 0
                curr_stone_color = board.EMPTY
                while 0 <= i < curr_state.num_rows and 0 <= j < curr_state.num_cols:
                    if curr_state[i, j] is board.EMPTY:
                        curr_stone_color = board.EMPTY
                        max_streak = 0
                    elif curr_stone_color is board.EMPTY:
                        curr_stone_color = curr_state[i, j]
                        max_streak = 1
                    elif curr_state[i, j] == curr_stone_color:
                        max_streak += 1
                    else:
                        max_streak = 1
                        curr_stone_color = curr_state[i, j]
                    if max_streak == 5:
                        return curr_stone_color
                    i += dir[0]
                    j += dir[1]
        return None

    def _search_ending_patterns(self, type, curr_state, stone_color):
        if type == 'w': # must win pattern
            patterns = [
                ((board.EMPTY, stone_color, stone_color, stone_color, stone_color), (0,)),
                ((stone_color, stone_color, stone_color, stone_color, board.EMPTY), (4,)),
            ]
        elif type == 'd' or type == 'a': # must defense/attack pattern
            patterns = [
                ((board.EMPTY, stone_color, stone_color, stone_color, stone_color), (0,)),
                ((stone_color, stone_color, stone_color, stone_color, board.EMPTY), (4,)),
                ((board.EMPTY, stone_color, stone_color, stone_color, board.EMPTY), (0, 4)),
            ]
        else:
            raise ValueError('type of pattern not recognized')

        for start_pos, dir in self.search_directions:
            for i, j in start_pos:
                while i < curr_state.num_rows and j < curr_state.num_cols:
                    for p, mv in patterns:
                        matched = True
                        for offset in range(len(p)):
                            x, y = i+dir[0]*offset, j+dir[1]*offset
                            if x >= self.num_rows or y >= self.num_cols or x < 0 or y < 0:
                                matched = False
                                break
                            if curr_state[x, y] != p[offset]:
                                matched = False
                                break
                        if matched:
                            return [(i+offset*dir[0], j+offset*dir[1]) for offset in mv]
                    i += 1
                    j += 1
        return None

    def next_states(self, state=None, next_player=None):
        """"""
        if state:
            curr_state = GomokuBoard(self.num_rows, self.num_cols, None, state)
        else:
            curr_state = self
        if next_player is None:
            prev_player = curr_state.current_player()
            if prev_player is board.PLAYER_A:
                next_player = board.PLAYER_B
            else:
                next_player = board.PLAYER_A
        if next_player is board.PLAYER_A:
            other_player = board.PLAYER_B
        else:
            other_player = board.PLAYER_A

        possible_moves = None
        possible_moves = self._search_ending_patterns('w', curr_state, next_player)
        if possible_moves is None:
            possible_moves = self._search_ending_patterns('d', curr_state, other_player)
        if possible_moves is None:
            possible_moves = self._search_ending_patterns('a', curr_state, next_player)
        if possible_moves is None:
            possible_moves = set() # to store element (idx_row, idx_col)
            for i in range(curr_state.num_rows):
                for j in range(curr_state.num_cols):
                    if curr_state[i, j] is not board.EMPTY:
                        for offset_i in range(-1, 2):
                            for offset_j in range(-1, 2):
                                pos_i = i + offset_i
                                pos_j = j + offset_j
                                if (0 <= pos_i < curr_state.num_rows) and (0 <= pos_j < curr_state.num_cols):
                                    if curr_state[pos_i, pos_j] is board.EMPTY:
                                        possible_moves.add((pos_i, pos_j))

        states = []
        for p in possible_moves:
            states.append(curr_state._serialize_potential_move(p, next_player))
        return states


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    grid_size = 15
    ticks = range(15)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid('on')
    g_board = GomokuBoard(grid_size, grid_size, board.PLAYER_A,
                          [(board.PLAYER_A,7,7)])
    winner = g_board.judge()
    g_next = g_board._serialize()

    g_next = random.choice(g_board.next_states(g_next))

    while not winner:
        g_next = random.choice(g_board.next_states(g_next))
        winner = g_board.judge(g_next)
        print GomokuBoard(grid_size, grid_size, None, g_next)
        print winner
        ax.imshow(GomokuBoard(grid_size, grid_size, None, g_next).convert_into_2d_array())
        plt.pause(0.25)
    plt.pause(10)

    array2d = GomokuBoard(grid_size, grid_size, None, g_next).convert_into_2d_array()
    for i in range(grid_size):
        for j in range(grid_size):
            if array2d[i,j] == 1:
                plt.scatter(i, j, c='r')
            elif array2d[i,j] == 2:
                plt.scatter(i, j, c='g')
