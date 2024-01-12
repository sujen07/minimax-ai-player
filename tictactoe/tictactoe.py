"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum([row.count('X') for row in board])
    o_count = sum([row.count('O') for row in board])
    if x_count <= o_count:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    all_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                all_actions.add((i,j))
    return all_actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]

    player_turn = player(board=board)
    all_actions = actions(board=board)
    if action not in all_actions:
        raise AssertionError("Invalid action")
    new_board = copy.deepcopy(board)
    new_board[i][j] = player_turn
    return new_board
    


    


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Rowwise Wins
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return row[0]

    # Columnal Wins
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return board[0][col]

    # Diagonal Wins
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if EMPTY not in board[0] + board[1] + board[2]:
        return True
    if winner(board):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    player_won = winner(board=board)
    if player_won == X:
        return 1
    elif player_won == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board=board):
        return None
    

    def optimize(board, alpha=float('-inf'), beta=float('inf')):
        if terminal(board=board):
            return utility(board=board), None
        
        player_turn = player(board=board)
        all_actions = actions(board=board)
        next_boards = [(result(board=board, action=action), action) for action in all_actions]
        if player_turn == X:
            value = float('-inf')
            for board, action in next_boards:
                score,_ = optimize(board=board, alpha=alpha, beta=beta)
                if score > value:
                    optimal_action = action
                    value = score
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, optimal_action
        else:
            value = float('inf')
            for board, action in next_boards:
                score,_ = optimize(board=board, alpha=alpha, beta=beta)
                if score < value:
                    optimal_action = action
                    value = score
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, optimal_action
    
    return optimize(board)[1]
        
        

