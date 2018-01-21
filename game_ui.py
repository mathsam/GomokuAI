import Tkinter
import tkMessageBox

import numpy as np

from board import PLAYER_A, PLAYER_B, TIE


class BoardGameCanvas(Tkinter.Canvas):

    result_text = {
        PLAYER_A: 'Player win',
        PLAYER_B: 'AI win',
        TIE: 'TIE',
    }

    def __init__(self, game_board, ai, parent=None):
        self.stone_num = 1
        self._cellsize = 60
        self._width = self._cellsize * game_board.num_cols + 40
        self._height = self._cellsize * game_board.num_rows + 40
        Tkinter.Canvas.__init__(self, parent, width=self._width, height=self._height, bg='#F5CBA7')
        self.game_board = game_board
        self.ai = ai

        if not tkMessageBox.askyesno(title='Please choose', message='Do you want to play black stone?'):
            self.result_text[PLAYER_A], self.result_text[PLAYER_B] =  (self.result_text[PLAYER_B],
                                                                       self.result_text[PLAYER_A])
            self.user_move_first = False
            ai_move = self.ai.best_move()
            self.ai.update_state(ai_move)
            self.game_board.update_state(ai_move)

        self.pack()
        self.draw_grid()
        self.draw_stones()
        self.bind('<ButtonPress-1>', self.user_move)


    def user_move(self, event):
        x, y = self.canvasx(event.x), self.canvasy(event.y)
        for j in range(len(self._col_grid)-1):
            if self._col_grid[j] < x < self._col_grid[j+1]:
                for i in range(len(self._row_grid)-1):
                    if self._row_grid[i] < y < self._row_grid[i+1]:
                        self.game_board.update_state((i, j))
                        self._draw_one_stone(i, j)
                        winner = self.game_board.judge()
                        if winner:
                            self.create_text(self._width/2, self._height/2,
                                             text=BoardGameCanvas.result_text[winner], font="Times 40 italic bold",
                                             fill="blue")
                            return
                        #print 'User state: DNN value %f' %self.ai._maintained_tree.value
                        self.ai.update_state((i, j))
                        ai_move = self.ai.best_move()
                        self.game_board.update_state(ai_move)
                        self.ai.update_state(ai_move)
                        self._draw_one_stone(*ai_move)
                        winner = self.game_board.judge()
                        if winner:
                            self.create_text(self._width/2, self._height/2,
                                             text=BoardGameCanvas.result_text[winner], font="Times 40 italic bold",
                                             fill="blue")
                            return

    def _draw_one_stone(self, row, col):
        stone_color = {PLAYER_A: 'black', PLAYER_B: 'white'}
        text_color = {PLAYER_A: 'yellow', PLAYER_B: 'green'}
        x0 = self._col_grid[col] + self._cellsize*0.1
        y0 = self._row_grid[row] + self._cellsize*0.1
        x1 = self._col_grid[col+1] - self._cellsize*0.1
        y1 = self._row_grid[row+1] - self._cellsize*0.1
        self.create_oval(x0, y0, x1, y1, fill=stone_color[self.game_board[row, col]])
        self.create_text((x0+x1)/2., (y0+y1)/2., text=str(self.stone_num), fill=text_color[self.game_board[row, col]])
        self.stone_num += 1
        self.update_idletasks()

    def draw_stones(self):
        for row, col in self.game_board.history:
            self._draw_one_stone(row, col)

    def draw_grid(self):
        left = 20
        right = self._width - 20
        lower = 20
        upper = self._height - 20
        self._row_grid = np.linspace(lower, upper, self.game_board.num_rows+1)
        self._col_grid = np.linspace(left, right, self.game_board.num_cols+1)
        for y in self._row_grid:
            self.create_line(left, y, right, y, fill='#5499C7', width=5)
        for x in self._col_grid:
            self.create_line(x, lower, x, upper, fill='#5499C7', width=5)


if __name__ == '__main__':
    import os
    from gomoku_board import GomokuBoard
    from ai_dnn import MCUCT_DNN
    top = Tkinter.Tk()
    #from ai import MCUCT
    gboard = GomokuBoard()
    base_dir = r'./dnn_data'
    training_status = eval(open(os.path.join(base_dir, 'training_status')).read())
    champion_model = os.path.join(base_dir, 'v%d' %training_status['current_champion'])
    ai = MCUCT_DNN(GomokuBoard, load_path=champion_model, min_num_sim=2**10)
    #ai = MCUCT(GomokuBoard, min_num_sim=1e4)
    app = BoardGameCanvas(gboard, ai, top)
    Tkinter.mainloop()
