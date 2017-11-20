import random
import time
from copy import deepcopy

import numpy as np

from board import TIE


class MCUCT(object):
    """
    Upper confidence bound applied to tree (UCT), a improved Monte Carlo Tree Search (MCTS)
    """
    # when no simulation is performed, the number of simulation is set to this number instead of 0
    _LAPLACE_SMOOTHING = 1e-3

    def __init__(self, cached_depth=2, wall_time=60, C=1.414):
        """
        The upper bound of the confidence is given by win_rate + C*sqrt(ln(n)/n_i)
        :param cached_depth: out of `cached_depth`, use simple Monte Carlo rather than UCT
        :param wall_time: time bound for running simulation
        :param C: parameter in the confidence bound
        """
        self.cached_depth = cached_depth
        self.wall_time = wall_time
        self.C = C

    def _explore(self):
        """Run one Monte Carlo simulation till whoever win and update simu_stats
        """
        curr_node = self.game_tree
        sim_path = []
        while curr_node.next_boards:
            if curr_node.is_UCT_ready:
                max_idx = curr_node.stats[:,2].argmax()
                curr_node = curr_node.next_boards[max_idx]
                sim_path.append(max_idx)
            else:
                curr_node_idx = random.randrange(len(curr_node.next_boards))
                curr_node = curr_node.next_boards[curr_node_idx]
                sim_path.append(curr_node_idx)
            if curr_node.judge() is not None:
                break

        curr_node = deepcopy(curr_node)
        while True:
            result = curr_node.judge()
            if result is not None:
                break
            random_next_move = random.choice(curr_node.avial_moves())
            curr_node.update_state(random_next_move)

        curr_node = self.game_tree
        for level, node_idx in enumerate(sim_path):
            if not curr_node.is_UCT_ready:
                if curr_node.stats[node_idx, 0] == 0:
                    curr_node._num_stat_ready += 1
                    curr_node.is_UCT_ready = (curr_node._num_stat_ready == len(curr_node.next_boards))
                    print 'Level %d ready' %(level+1)
            curr_node._total_num_sim += 1
            curr_node.stats[node_idx, 0] += 1
            if result == TIE:
                curr_node.stats[node_idx, 1] += 0.5
            elif result == curr_node.current_player():
                curr_node.stats[node_idx, 1] += 1
            curr_node.stats[node_idx, 2] = (
                curr_node.stats[node_idx, 1]/curr_node.stats[node_idx, 0] +
                self.C * np.sqrt(np.log(curr_node._total_num_sim) / curr_node.stats[node_idx, 0])
            )

    def _initialize_cached_tree(self, board, current_level=0):
        if current_level == self.cached_depth:
            board.next_boards = None
            return
        moves = board.avial_moves()
        next_boards = []
        for m in moves:
            child = deepcopy(board)
            child.update_state(m)
            self._initialize_cached_tree(child, current_level+1)
            next_boards.append(child)
        board.next_boards = next_boards
        # (total_simu_num, win_num, win_rate_upper_bound)
        board.stats = np.zeros((len(next_boards), 3), dtype=np.float32)
        board.stats[:, 0] = MCUCT._LAPLACE_SMOOTHING
        board._num_stat_ready = 0  # how many number out of len(next_boards) has been simulated
        board._total_num_sim = 0
        board.is_UCT_ready = False  # if stats for all the next_boards are ready so that UCT can kick in
        return

    def best_move(self, board):
        self.current_player = board.current_player()
        self.game_tree = deepcopy(board)
        self._initialize_cached_tree(self.game_tree)

        start_time = time.time()
        while time.time() - start_time < self.wall_time:
            self._explore()

        winning_rate = self.game_tree.stats[:,1]/self.game_tree.stats[:,0]
        best_move_idx = winning_rate.argmax()
        print 'Best move, total play: %f, win play: %f, win rate %f' %tuple(self.game_tree.stats[best_move_idx,:])
        print 'Total simulation %d' %self.game_tree._total_num_sim
        return self.game_tree.next_boards[best_move_idx].last_move


if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    g_board = GomokuBoard()
    g_board.update_state((4, 4))
    mc = MCUCT(cached_depth=1, wall_time=300)
    print mc.best_move(g_board)
