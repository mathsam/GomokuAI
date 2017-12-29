from copy import deepcopy

import numpy as np

from board import Board, TIE, PLAYER_A
from dnn import AINet


class MCUCT_DNN(object):

    # need to maintain context in each Python instance when run in parallel using ipyparallel
    _maintained_tree = {}
    _params = {}
    _LAPLACE_SMOOTHING = 1.0

    def __init__(self, board_constructor, C=1, min_num_sim=2**12):
        self.dnn = AINet('restart')
        self.C = C
        self.min_num_sim = min_num_sim
        self._maintained_tree = board_constructor()
        self._initialize_tree_node(self._maintained_tree)

    def update_state(self, move):
        which_child = self._maintained_tree.avial_moves().index(move)
        child = self._maintained_tree.children[which_child]
        if not isinstance(child, Board):
            child = self._maintained_tree.copy()
            child.update_state(move)
            self._initialize_tree_node(child)
        self._maintained_tree = child

    def best_move(self):
        for i in xrange(self.min_num_sim):
            self._explore()
        best_move_idx = self._maintained_tree.stats[:, 0].argmax()
        return self._maintained_tree.avial_moves()[best_move_idx]

    def _explore(self):
        """Run one Monte Carlo simulation till whoever win and update simu_stats
        """
        root = self._maintained_tree
        C = self.C
        curr_node = root
        sim_path = []
        while True:
            if curr_node.judge() is not None: # if leaf node
                leaf_node = curr_node
                break
            next_node_idx = curr_node.stats[:, 2].argmax()
            sim_path.append(next_node_idx)
            child_node = curr_node.children[next_node_idx]
            if not isinstance(child_node, Board):
                new_node = curr_node.copy()
                new_node.update_state(child_node)
                self._initialize_tree_node(new_node)
                curr_node.children[next_node_idx] = new_node
                leaf_node = new_node
                break
            curr_node = child_node

        curr_node = root
        for node_idx in sim_path:
            curr_node._total_num_sim += 1
            curr_node.stats[node_idx, 0] += 1
            if leaf_node.current_player() == curr_node.current_player():
                curr_node.stats[node_idx, 1] += 1+leaf_node.value
            else:
                curr_node.stats[node_idx, 1] -= leaf_node.value
            curr_node.stats[:, 2] = (
                curr_node.stats[:, 1] / curr_node.stats[:, 0] +
                C * np.sqrt(curr_node._total_num_sim) / curr_node.stats[:, 0] * curr_node.prior_move_prob[:]
            )
            curr_node = curr_node.children[node_idx]

    def _initialize_tree_node(self, node):
        node.children = deepcopy(node.avial_moves())
        if node.judge() is not None:
            if node.judge() == TIE:
                node.value = 0
            else:  # current player is already lost
                node.value = -1
            return

        board_state = node.convert_into_2d_array(dtype=np.float32)[..., np.newaxis]
        if node.current_player() != PLAYER_A:
            board_state = -board_state
        move, val = self.dnn.pred(board_state)
        node.value = val
        move = move.reshape((Board.num_rows, Board.num_cols))
        node.prior_move_prob = [move[idx] for idx in node.avial_moves()]

        node.stats = np.zeros((len(node.children), 3), dtype=np.float32) # (num_sim_this_node, cumulated value, upper_win_rate_bound)
        node.stats[:, 2] = node.prior_move_prob
        node.stats[:, 0] = self._LAPLACE_SMOOTHING
        node._total_num_sim = 0


if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    ai = MCUCT_DNN(GomokuBoard, min_num_sim=1)
    ai.update_state((4, 4))
    print ai.best_move()
