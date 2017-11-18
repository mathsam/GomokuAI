import numpy as np
import random
import time
import board


class MCUCT(object):
    """
    Upper confidence bound applied to tree (UCT), a improved Monte Carlo Tree Search (MCTS)
    """

    history = {
        board.PLAYER_A: [],
        board.PLAYER_B: [],
    }

    def __init__(self, board, wall_time=60, C=1.414):
        """
        The upper bound of the confidence is given by win_rate + C*sqrt(ln(n)/n_i)
        :param board: _board with current state
        :param wall_time: time bound for running simulation
        :param C: parameter in the confidence bound
        """
        self.board = board
        self.wall_time = wall_time
        self.C = C
        # record number of plays and wins for each explored state, value is tuple (num_wins, num_plays)
        self.simu_stats = dict()
        self.next_states = board.next_states()

    def _explore(self):
        """Run one Monte Carlo simulation till whoever win and update simu_stats
        """
        curr_simu_path = []
        next_states = self.next_states
        while True:
            if all([s in self.simu_stats for s in next_states]):
                stats = np.array([self.simu_stats[s] for s in next_states], dtype=np.float32)
                total_num_plays = stats[:,0].sum()
                upper_confidence_bound = (stats[:,1]/stats[:,0] +
                                          self.C * np.sqrt(np.log(total_num_plays) / stats[:,0]))
                max_idx = upper_confidence_bound.argmax()
                chosen_state = next_states[max_idx]
            else:
                chosen_state = random.choice(next_states)

            curr_simu_path.append(chosen_state)

            result = self.board.judge(chosen_state)
            if result:
                break
            next_states = self.board.next_states(chosen_state)

        for s in curr_simu_path:
            if s not in self.simu_stats:
                self.simu_stats[s] = [1, 0]
            else:
                self.simu_stats[s][0] += 1
            if result == self.board.current_player(s):
                self.simu_stats[s][1] += 1
            elif result == board.TIE:
                self.simu_stats[s][1] += 0.5

    def best_move(self):
        start_time = time.time()
        while time.time() - start_time < self.wall_time:
            self._explore()
        winning_rate = 0.
        total_plays = 0
        for s in self.next_states:
            if s in self.simu_stats:
                total_plays += self.simu_stats[s][0]
                curr_winnning_rate = self.simu_stats[s][1]/float(self.simu_stats[s][0])
                if curr_winnning_rate >= winning_rate:
                    best_s = s
                    winning_rate = curr_winnning_rate
        MCUCT.history[self.board.current_player(best_s)] = winning_rate
        print self.board.current_player(best_s)
        print 'best winning rate', winning_rate
        print '%d / %d' %tuple(self.simu_stats[best_s])
        print 'total plays', total_plays
        return best_s



