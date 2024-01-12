
# Minimax AI Player

This repository contains two separate implementations of the Minimax algorithm in games: Chess and Tic-Tac-Toe. Both games are interactive and allow the user to play against an AI opponent powered by the Minimax algorithm.

## Contents

- `chess/`
  - `requirements.txt`
  - `main.py`
- `tictactoe/`
  - `requirements.txt`
  - `main.py`

## Installation

To run these games, you need to have Python installed on your system. Additionally, each game has its own set of dependencies which can be installed using `pip`.

### Chess

Navigate to the `chess` directory and install the requirements:

```bash
cd chess
pip install -r requirements.txt

Usage
To play the games, run the main.py file in the respective game directory.

Chess
bash
Copy code
cd chess
python main.py
Tic-Tac-Toe
bash
Copy code
cd tictactoe
python main.py
Minimax Algorithm
The Minimax algorithm is a decision-making AI algorithm used in decision making and game theory. It provides an optimal move for the player assuming that the opponent is also playing optimally. In a zero-sum game, which means one player's gain is another's loss, the Minimax algorithm helps to minimize the possible loss for a worst-case scenario.

When dealing with games like chess and tic-tac-toe, the algorithm considers all possible moves, evaluates them, and then chooses the best move. The AI looks ahead and tries to minimize the potential loss in the worst-case scenario.

Screenshots
Chess

Tic-Tac-Toe

Contributing
Contributions to this project are welcome. Please ensure to update tests as appropriate.

License
MIT
