from board import Board
from board import PLAYER_A, PLAYER_B, TIE


class GomokuBoard(Board):

    search_directions = (
        (0, 1), (1, 0), (1, 1), (1, -1),
    )

    def __init__(self):
        Board.__init__(self)
        self.game_result = None

    def update_state(self, move):
        Board.update_state(self, move)
        self.game_result = self._judge(move)
        if self.game_result is not None:
            self._empty_indices = []

    def judge(self):
        return self.game_result

    def _judge(self, move):
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
                    i = move[0] + dir[0]*offset*sign
                    j = move[1] + dir[1]*offset*sign
                    if 0 <= i < Board.num_rows and 0 <= j < Board.num_cols and self[i, j] == last_color:
                        max_stone_in_line += 1
                    else:
                        break
            if max_stone_in_line >= 5:
                self.judge = lambda : self.last_player
                return self.last_player
        return None


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
