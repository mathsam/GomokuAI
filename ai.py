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
    _LAPLACE_SMOOTHING_UPPER_BOUND = _LAPLACE_SMOOTHING + 1e-5 # Used for float number comparision

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
        for i in range(self.cached_depth):
            if curr_node.is_UCT_ready:
                curr_node_idx = curr_node.stats[:, 2].argmax()
            else:
                curr_node_idx = random.randrange(len(curr_node.next_boards))
            sim_path.append(curr_node_idx)
            child_node = curr_node.next_boards[curr_node_idx]
            if isinstance(child_node, tuple):
                child_node = deepcopy(curr_node)
                child_node.update_state(curr_node.next_boards[curr_node_idx])
                self._initialize_cached_tree(child_node)
                curr_node.next_boards[curr_node_idx] = child_node
            curr_node = child_node
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
                if curr_node.stats[node_idx, 0] <= MCUCT._LAPLACE_SMOOTHING_UPPER_BOUND:
                    curr_node._num_stat_ready += 1
                    curr_node.is_UCT_ready = (curr_node._num_stat_ready == len(curr_node.next_boards))
                    if curr_node.is_UCT_ready:
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
            curr_node = curr_node.next_boards[node_idx]

    def _initialize_cached_tree(self, node):
        node.next_boards = deepcopy(node.avial_moves())
        node.stats = np.zeros((len(node.next_boards), 3), dtype=np.float32)
        node.stats[:, 0] = MCUCT._LAPLACE_SMOOTHING
        node._num_stat_ready = 0 # how many number of len(next_boards) has been simulated
        node._total_num_sim = 0
        node.is_UCT_ready = False # if all nodes have been simulated, UCT can kick in

    def best_move(self, board):
        self.current_player = board.current_player()
        self.game_tree = deepcopy(board)
        self._initialize_cached_tree(self.game_tree)

        start_time = time.time()
        while time.time() - start_time < self.wall_time:
            self._explore()

        winning_rate = self.game_tree.stats[:,1]/self.game_tree.stats[:,0]
        best_move_idx = winning_rate.argmax()
        print self.game_tree.current_player()
        print 'Best move, total play: %f, win play: %f, win rate %f' %tuple(self.game_tree.stats[best_move_idx,:])
        print 'Total simulation %d' %self.game_tree._total_num_sim
        return self.game_tree.next_boards[best_move_idx].last_move


if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    g_board = GomokuBoard()
    g_board.update_state((4, 4))
    mc = MCUCT(cached_depth=1, wall_time=300)
    print mc.best_move(g_board)
