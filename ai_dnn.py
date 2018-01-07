from copy import deepcopy

import numpy as np
from numpy.random import choice

from board import Board, TIE, PLAYER_A
from dnn import AINet


def top_k_moves(prob, k):
    topk_idx = sorted(range(len(prob)), key=lambda i: prob[i])[-k:]
    topk_prob = prob[topk_idx]
    topk_prob /= topk_prob.sum()
    return topk_idx, topk_prob


class MCUCT_DNN(object):

    # need to maintain context in each Python instance when run in parallel using ipyparallel
    _params = {}
    _LAPLACE_SMOOTHING = 1.0

    def __init__(self, board_constructor, load_path=r'./dnn_data/champion/', C=10, min_num_sim=2**12,
                 training_mode=False):
        min_num_sim = int(min_num_sim)
        self.dnn = AINet('restart', load_path=load_path, use_gpu=False)
        self.C = C
        self.min_num_sim = min_num_sim
        self.game_board = board_constructor()
        self._initialize_tree_node(self.game_board)
        self.training_mode = training_mode

    def update_state(self, move):
        which_child = self.game_board.avial_moves().index(move)
        child = self.game_board.children[which_child]
        if not isinstance(child, Board):
            child = self.game_board.copy()
            child.update_state(move)
            self._initialize_tree_node(child)
        self.game_board = child

    def best_move(self):
        self.max_depth = 1
        for i in xrange(self.min_num_sim):
            self._explore()
        if not self.training_mode:
            best_move_idx = self.game_board.stats[:, 0].argmax()
        else:
            top_moves, top_prob = top_k_moves(self.game_board.stats[:, 0], 5)
            already_win = False
            for potential_move in top_moves:
                if (isinstance(self.game_board.children[potential_move], Board) and
                self.game_board.children[potential_move].judge() is not None):
                    best_move_idx = potential_move
                    already_win = True
                    break
            if not already_win:
                best_move_idx = choice(top_moves, p=top_prob)
                print 'Move prob', top_prob
        print 'AI move: DNN value %f, MCTS value %f' %(self.game_board.value,
                                                       self.game_board.stats[best_move_idx, 2])
        print 'max search depth: %d' %self.max_depth
        self.game_stats = self.game_board.stats[:, 0]
        return self.game_board.avial_moves()[best_move_idx]

    def _explore(self):
        """Run one Monte Carlo simulation till whoever win and update simu_stats
        """
        root = self.game_board
        C = self.C
        curr_node = root
        sim_path = []
        while True:
            if curr_node.judge() is not None: # if leaf node
                leaf_node = curr_node
                break
            next_node_idx = curr_node.stats[:, 2].argmax()
            #next_node_idx = choice(len(curr_node.stats[:, 2]), p=curr_node.stats[:, 2])
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

        self.max_depth = max(self.max_depth, len(sim_path))
        curr_node = root
        for node_idx in sim_path:
            curr_node._total_num_sim += 1
            curr_node.stats[node_idx, 0] += 1
            if leaf_node.current_player() == curr_node.current_player():
                curr_node.stats[node_idx, 1] += leaf_node.value
            else:
                curr_node.stats[node_idx, 1] -= leaf_node.value
            curr_node.stats[:, 2] = (
                curr_node.stats[:, 1] / curr_node.stats[:, 0] +
                C * np.sqrt(curr_node._total_num_sim) / curr_node.stats[:, 0] * curr_node.prior_move_prob[:]
            )
            # min_stats = curr_node.stats[:, 2].min()
            # curr_node.stats[:, 2] -= min_stats
            # curr_node.stats[:, 2] /= curr_node.stats[:, 2].sum()
            curr_node = curr_node.children[node_idx]

    def _initialize_tree_node(self, node):
        node.children = deepcopy(node.avial_moves())
        if node.judge() is not None:
            if node.judge() == TIE:
                node.value = 0
            else:  # current player is already lost
                node.value = -4
            return

        board_state = node.convert_into_2d_array(dtype=np.float32)[..., np.newaxis]
        if node.current_player() != PLAYER_A:
            board_state = -board_state
        move, val = self.dnn.pred(board_state)
        node.value = val
        move = move.reshape((Board.num_rows, Board.num_cols))
        node.prior_move_prob = np.array([move[idx] for idx in node.avial_moves()])

        node.stats = np.zeros((len(node.children), 3), dtype=np.float32) # (num_sim_this_node, cumulated value, upper_win_rate_bound)
        node.stats[:, 2] = node.prior_move_prob/node.prior_move_prob.sum()
        node.stats[:, 0] = self._LAPLACE_SMOOTHING
        node._total_num_sim = 0


if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    ai = MCUCT_DNN(GomokuBoard, min_num_sim=1)
    ai.update_state((4, 4))
    print ai.best_move()
