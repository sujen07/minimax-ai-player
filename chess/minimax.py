import chess
import random

def terminal(board):
    if board.is_checkmate() or board.is_stalemate():
        return True
    return False

def result(board, move):
    # Copy the original board
    new_board = board.copy()

    # Apply the move to the new board
    new_board.push(move)

    return new_board


CHECKMATE_SCORE = 10000  # A large value for checkmate

def evaluate_board(board):
    # Check for game-ending conditions
    if board.is_checkmate():
        # If it's checkmate, check who is the winner
        if board.turn == chess.WHITE:
            return -CHECKMATE_SCORE  # White is checkmated, bad for White
        else:
            return CHECKMATE_SCORE   # Black is checkmated, good for White
    elif board.is_stalemate() or board.is_insufficient_material():
        # Draw conditions
        return 0

    # Basic material count evaluation for non-terminal states
    score = 0
    for (piece, value) in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                           (chess.ROOK, 5), (chess.QUEEN, 9)]:
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value

    # Normalize the score to be between -1 and 1
    # Note: This normalization is optional and depends on your specific use case
    normalized_score = score / CHECKMATE_SCORE
    return max(min(normalized_score, 1), -1)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board=board):
        return None
    

    def optimize(board, depth=3, alpha=float('-inf'), beta=float('inf')):
        if terminal(board=board) or depth <= 0:
            return evaluate_board(board=board), None
        
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)

        if board.turn == chess.WHITE:  # Maximizing player
            value = float('-inf')
            optimal_action = None

            for action in legal_moves:
                new_board = board.copy()
                new_board.push(action)
                score, _ = optimize(new_board, depth -1, alpha, beta)
                if score > value:
                    value = score
                    optimal_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, optimal_action
        else:  # Minimizing player
            value = float('inf')
            optimal_action = None
            for action in legal_moves:
                new_board = board.copy()
                new_board.push(action)
                score, _ = optimize(new_board, depth - 1, alpha, beta)
                if score < value:
                    value = score
                    optimal_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, optimal_action

    
    return optimize(board)[1]
