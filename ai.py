import numpy as np

from para_search import TreeSearch


class MCUCT(object):
    """
    Upper confidence bound applied to tree (UCT), a improved Monte Carlo Tree Search (MCTS)
    """
    _ai_uid = 0

    def __init__(self, board_constructor, run_type='ipyparallel'):
        """
        The upper bound of the confidence is given by win_rate + C*sqrt(ln(n)/n_i)
        """
        self.game_board = board_constructor()
        self.uid = MCUCT._ai_uid
        self.run_type = run_type
        if run_type != 'ipyparallel':
            TreeSearch.init_tree(self.uid, board_constructor)
        else:
            self._init_parallel_context(board_constructor)
        MCUCT._ai_uid += 1

    def update_state(self, move):
        if self.run_type == 'ipyparallel':
            self._update_state_parallel(move)
        else:
            self._update_state_single(move)

    def best_move(self):
        if self.run_type == 'ipyparallel':
            return self._best_move_parallel()
        else:
            return self._best_move_single()

    def _init_parallel_context(self, board_constructor):
        import ipyparallel as ipp
        self.workers = ipp.Client()
        for worker_id in self.workers.ids:
            r = self.workers[worker_id].apply_async(TreeSearch.init_tree, self.uid, board_constructor)
        self.workers.wait()

    def _update_state_parallel(self, move):
        self.game_board.update_state(move)
        for worker_id in self.workers.ids:
            r = self.workers[worker_id].apply_async(TreeSearch.update_state, self.uid, move)
        self.workers.wait()

    def _update_state_single(self, move):
        TreeSearch.update_state(self.uid, move)
        self.game_board.update_state(move)

    def _best_move_parallel(self):
        avial_moves = self.game_board.avial_moves()
        stats_all_workers = []
        for worker_id in self.workers.ids:
            stats_all_workers.append(
                self.workers[worker_id].apply_async(TreeSearch.next_move_stats, self.uid))
        self.workers.wait_interactive()
        stats = np.zeros(len(avial_moves))
        for r in stats_all_workers:
            stats += r.get()
        print 'total sim', stats.sum()
        best_move_idx = stats.argmax()
        return avial_moves[best_move_idx]

    def _best_move_single(self):
        stats = TreeSearch.next_move_stats(self.uid)
        best_move_idx = stats.argmax()
        return self.game_board.avial_moves()[best_move_idx]

    def __del__(self):
        if self.run_type == 'ipyparallel':
            for worker_id in self.workers.ids:
                self.workers[worker_id].apply_sync(TreeSearch.destroy_tree, (self.uid,))

if __name__ == '__main__':

    from gomoku_board import GomokuBoard
    ai = MCUCT(GomokuBoard)
    ai.update_state((4, 4))
    print ai.best_move()
