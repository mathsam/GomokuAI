import Tkinter

import numpy as np

from board import PLAYER_A, PLAYER_B


class BoardGameCanvas(Tkinter.Canvas):

    def __init__(self, game_board, parent=None):
        self._cellsize = 60
        self._width = self._cellsize * game_board.num_cols + 40
        self._height = self._cellsize * game_board.num_rows + 40
        Tkinter.Canvas.__init__(self, parent, width=self._width, height=self._height, bg='#F5CBA7')
        self.game_board = game_board

        self.pack()
        self.draw_grid()
        self.draw_stones()

        self.bind('<ButtonPress-1>', self.user_move)

    def user_move(self, event):
        x, y = self.canvasx(event.x), self.canvasy(event.y)
        print x, y
        for j in range(len(self._col_grid)-1):
            if self._col_grid[j] < x < self._col_grid[j+1]:
                for i in range(len(self._row_grid)-1):
                    if self._row_grid[i] < y < self._row_grid[i+1]:
                        self.game_board.update_state((i, j))
                        self._draw_one_stone(i, j)
                        return

    def _draw_one_stone(self, row, col):
        stone_color = {PLAYER_A: 'black', PLAYER_B: 'white'}
        x0 = self._col_grid[col] + self._cellsize*0.1
        y0 = self._row_grid[row] + self._cellsize*0.1
        x1 = self._col_grid[col+1] - self._cellsize*0.1
        y1 = self._row_grid[row+1] - self._cellsize*0.1
        self.create_oval(x0, y0, x1, y1, fill=stone_color[self.game_board[row, col]])

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
    top = Tkinter.Tk()
    from gomoku_board import Board
    gboard = Board()
    gboard.update_state((4,5))
    gboard.update_state((5,6))
    app = BoardGameCanvas(gboard, top)
    Tkinter.mainloop()
