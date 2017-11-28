import random
import time
from copy import deepcopy

import numpy as np

from board import Board


class TreeSearch(object):

    # need to maintain context in each Python instance when run in parallel using ipyparallel
    _maintained_tree = {}
    _params = {}
    _LAPLACE_SMOOTHING = 1e-1

    @staticmethod
    def init_tree(uid, board_constructor, C):
        """initialize a root node with constructor `board_constructor` and store in in
        _maintained_tree with key `uid`"""
        TreeSearch._maintained_tree[uid] = board_constructor()
        TreeSearch._initialize_tree_node(TreeSearch._maintained_tree[uid])
        TreeSearch._params[uid] = {'C': C}

    @staticmethod
    def update_state(uid, move):
        root = TreeSearch._maintained_tree[uid]
        which_child = root.avial_moves().index(move)
        child = root.children[which_child]
        if not isinstance(child, Board):
            child = root
            child.update_state(move)
            TreeSearch._initialize_tree_node(child)
        TreeSearch._maintained_tree[uid] = child

    @staticmethod
    def destroy_tree(uid):
        TreeSearch._maintained_tree.pop(uid)
        TreeSearch._params.pop(uid)

    @staticmethod
    def next_move_stats(uid):
        C = TreeSearch._params[uid]['C']
        root = TreeSearch._maintained_tree[uid]
        start_time = time.time()
        while time.time() - start_time < 10:
            TreeSearch._explore(root, C)
        return root.stats[:, 0].flatten()

    @staticmethod
    def _random_playout(node):
        node = deepcopy(node)
        while True:
            result = node.judge()
            if result is not None:
                break
            random_next_move = random.choice(node.avial_moves())
            node.update_state(random_next_move)
        return result

    @staticmethod
    def _explore(root, C):
        """Run one Monte Carlo simulation till whoever win and update simu_stats
        """
        curr_node = root
        sim_path = []
        while True:
            if not curr_node.stats.size: # if leaf node
                result = curr_node.judge()
                break
            next_node_idx = curr_node.stats[:, 2].argmax()
            sim_path.append(next_node_idx)
            child_node = curr_node.children[next_node_idx]
            if not isinstance(child_node, Board):
                new_node = deepcopy(curr_node)
                new_node.update_state(child_node)
                TreeSearch._initialize_tree_node(new_node)
                result = TreeSearch._random_playout(new_node)
                curr_node.children[next_node_idx] = new_node
                break
            curr_node = child_node

        curr_node = root
        for node_idx in sim_path:
            curr_node._total_num_sim += 1
            curr_node.stats[node_idx, 0] += 1
            if result == curr_node.current_player():
                curr_node.stats[node_idx, 1] += 1
            curr_node.stats[node_idx, 2] = (
                curr_node.stats[node_idx, 1]/curr_node.stats[node_idx, 0] +
                C * np.sqrt(np.log(curr_node._total_num_sim) / curr_node.stats[node_idx, 0])
            )
            curr_node = curr_node.children[node_idx]
        #curr_node._total_num_sim += 1

    @staticmethod
    def _initialize_tree_node(node):
        node.children = deepcopy(node.avial_moves())
        node.stats = np.zeros((len(node.children), 3), dtype=np.float32) # (num_sim_this_node, num_win, upper_win_rate_bound)
        node.stats[:, 0:2] = TreeSearch._LAPLACE_SMOOTHING
        node.stats[:, 2] = np.random.uniform(1000, 10000, len(node.children))  # play unvisited move at least once and in a random order
        node._total_num_sim = 0
