import time

import numpy as np

from para_search import TreeSearch


class MCUCT(object):
    """
    Upper confidence bound applied to tree (UCT), a improved Monte Carlo Tree Search (MCTS)
    """
    _ai_uid = 0

    def __init__(self, board_constructor, C=0.3, run_type='ipyparallel', min_num_sim=6e4):
        """
        The upper bound of the confidence is given by win_rate + C*sqrt(ln(n)/n_i)
        """
        self.C = C
        self.min_num_sim = min_num_sim
        self.game_board = board_constructor()
        self.uid = MCUCT._ai_uid
        self.run_type = run_type
        if run_type != 'ipyparallel':
            TreeSearch.init_tree(self.uid, board_constructor, self.C)
        else:
            self._init_parallel_context(board_constructor)
        #MCUCT._ai_uid += 1

    def update_state(self, move):
        if self.run_type == 'ipyparallel':
            self._update_state_parallel(move)
        else:
            self._update_state_single(move)

    def best_move(self):
        start_time = time.time()
        if self.run_type == 'ipyparallel':
            result = self._best_move_parallel()
        else:
            result = self._best_move_single()
        print 'time spent', time.time() - start_time
        return result

    def _init_parallel_context(self, board_constructor):
        import ipyparallel as ipp
        self.workers = ipp.Client()
        self.workers.purge_everything()
        for worker_id in self.workers.ids:
            self.workers[worker_id].apply_async(
                TreeSearch.init_tree, self.uid, board_constructor, self.C)
        self.workers.wait()

    def _update_state_parallel(self, move):
        self.game_board.update_state(move)
        for worker_id in self.workers.ids:
            self.workers[worker_id].apply_async(TreeSearch.update_state, self.uid, move)
        self.workers.wait()

    def _update_state_single(self, move):
        TreeSearch.update_state(self.uid, move)
        self.game_board.update_state(move)

    def _best_move_parallel(self):
        avial_moves = self.game_board.avial_moves()
        while True:
            stats_all_workers = []
            for worker_id in self.workers.ids:
                stats_all_workers.append(
                    self.workers[worker_id].apply_async(TreeSearch.next_move_stats, self.uid))
            self.workers.wait()
            stats = np.zeros(len(avial_moves))
            for r in stats_all_workers:
                stats += r.get()
            if stats.sum() > self.min_num_sim:
                print 'total sim', stats.sum()
                break
        best_move_idx = stats.argmax()
        self.game_stats = stats
        return avial_moves[best_move_idx]

    def _best_move_single(self):
        while True:
            stats = TreeSearch.next_move_stats(self.uid)
            if stats.sum() > self.min_num_sim:
                break
        best_move_idx = stats.argmax()
        self.game_stats = stats
        return self.game_board.avial_moves()[best_move_idx]

    def __del__(self):
        if self.run_type == 'ipyparallel':
            for worker_id in self.workers.ids:
                self.workers[worker_id].apply_sync(TreeSearch.destroy_tree, self.uid)

if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    ai = MCUCT(GomokuBoard, min_num_sim=24e4)
    ai.update_state((4, 4))
    print ai.best_move()
