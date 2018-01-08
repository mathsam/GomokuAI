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


## References

  * Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270
  * Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587), 484–489. https://doi.org/10.1038/nature16961
  * Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., … Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236
  * Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning, 282–293. https://doi.org/10.1007/11871842_29
  * Chaslot, G. M. J. B., Winands, M. H. M., & Van Den Herik, H. J. (2008). Parallel Monte-Carlo tree search. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 5131 LNCS, 60–71. https://doi.org/10.1007/978-3-540-87608-3_6
