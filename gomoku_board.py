from board import Board
from board import PLAYER_A, PLAYER_B, TIE


class GomokuBoard(Board):

    search_directions = (
        (0, 1), (1, 0), (1, 1), (1, -1),
    )

    def __init__(self, initial_state=None):
        Board.__init__(self, initial_state)
        # limit return of avial_moves to be within the boundaries to reduce search space
        # (left, right, top, bottom)
        if initial_state is None:
            self._explore_boundaries = [float('inf'), float('-inf'), float('inf'), float('-inf')]

    def update_state(self, move):
        Board.update_state(self, move)
        self._explore_boundaries[0] = min(self._explore_boundaries[0], move[0]-2)
        self._explore_boundaries[1] = max(self._explore_boundaries[1], move[0]+2)
        self._explore_boundaries[2] = min(self._explore_boundaries[2], move[1]-2)
        self._explore_boundaries[3] = max(self._explore_boundaries[3], move[1]+2)

    def judge(self, state=None):
        """Only need to search if the last move forms a five in a line
        :param state: None(default)|str(serialized state)
        :return: Return the player that wins if exists, otherwise return None
        """
        if self.num_stones == Board.num_rows*Board.num_cols:
            return TIE
        last_color = self.last_player
        for dir in GomokuBoard.search_directions:
            max_stone_in_line = 1
            for sign in [-1, 1]:
                for offset in xrange(1, 5):
                    i = self.last_move[0] + dir[0]*offset*sign
                    j = self.last_move[1] + dir[1]*offset*sign
                    if 0 <= i < Board.num_rows and 0 <= j < Board.num_cols and self[i, j] == last_color:
                        max_stone_in_line += 1
                    else:
                        break
            if max_stone_in_line >= 5:
                return self.last_player
        return None

    def avial_moves(self):
        empty_indices = self._empty_indices
        filtered_empty_indices = [(i, j) for i, j in empty_indices if
                                  (self._explore_boundaries[0] <= i <= self._explore_boundaries[1]) and
                                  (self._explore_boundaries[2] <= j <= self._explore_boundaries[3])]
        return filtered_empty_indices


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    grid_size = 9
    g_board = GomokuBoard()
    g_board.update_state((grid_size//2, grid_size//2))
    winner = g_board.judge()

    while not winner:
        next_move = random.choice(g_board.avial_moves())
        g_board.update_state(next_move)
        winner = g_board.judge()
        print g_board
        #g_board.draw(ax)
        #plt.pause(0.5)

    if winner == PLAYER_A:
        print 'Winner is PLAYER A'
    elif winner == PLAYER_B:
        print 'Winner is PLAYER B'
    else:
        print 'Match TIE'
    print g_board
    g_board.draw()

    plt.figure()
    plt.axis('equal')
    plt.grid('on')
    plt.xlim([0, 10])
    plt.xlim([0, 10])
    array2d = g_board.convert_into_2d_array()
    for i in range(grid_size):
        for j in range(grid_size):
            if array2d[i,j] == 1:
                plt.scatter(i+1, j+1, c='r', s=30)
            elif array2d[i,j] == 2:
                plt.scatter(i+1, j+1, c='g', s=30)
