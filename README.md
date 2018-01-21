# GomokuAI

Implement an AI to play Gomoku (Five in a Row) board game using method described by the Alpha Go/Zero papers.

Algorithm is Monte Carlo Tree Search (MCTS) guided by neural network. Reinforcement learning with selfplay is carried out to strengthen the neural network.

## Components

`ai.py` implements a pure MCTS algorithm. Random playouts are simulated with multi-armed bandit method to guide the exploitation.

`dnn.py` constructs the convolutional neural network (CNN).

`ai_dnn.py` uses the CNN to guide the MCTS.

`game_ui.py` provides a GUI for user to play with AI.

## Dependency

Tensorflow

## Start to play with AI

`python game_ui.py`

![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/game_ui.png)

## Learn to play games itself by self-play

On "day 0", the neural network is randomly initialized, and therefore the AI plays randomly. A typical self-play game looks like

![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/day0.png)

About 3000 games are played by AI against itself. These games are then used to train the neural networks: the value network
that predicts the winning probability at current state, and a policy network that select the next best move. After this
 the first training, the game played by AI no longer resembles random move but makes much more sense. A typical "day 1"
 game looks like

 ![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/day1.png)

 It is amazing that the AI can actually learn something from the seemingly random "day 0" self-plays.

 Then the "day 1" AI generates more self-plays which are used to train "day 2" AI, and so on and so forth. At day 6,
 the AI is already able to beat me. Here's in middle of the game, AI is playing black stone. I had a good winning chance
 if AI didn't play 15 but I play 15, and then I play between 4 and 14 and I'll win. However, AI seems to be able to look
 two steps forward and blocked my best move.

 ![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/v6_AI_block_my_winning_move.png)

 In the end of the game, AI played a 4x5 with move 27 and ended the game.

 ![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/v6_AI_beat_me.png)

 The progression of Elo score during reinforcement training is shown below:

 ![alt text](https://github.com/mathsam/GomokuAI/blob/master/analysis/elo_score_progression.png)

 It increases quickly in the beginning, and then slower and slower and eventually saturated. Possible way to future
 improve AI is to make the neural network deeper. The current architecture only 3 convolutional layers with 1 fully
 connected layers. As a comparison, Alpha Zero has 17 residual layers.


## References

  * Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270
  * Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587), 484–489. https://doi.org/10.1038/nature16961
  * Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., … Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236
  * Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning, 282–293. https://doi.org/10.1007/11871842_29
  * Chaslot, G. M. J. B., Winands, M. H. M., & Van Den Herik, H. J. (2008). Parallel Monte-Carlo tree search. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 5131 LNCS, 60–71. https://doi.org/10.1007/978-3-540-87608-3_6
