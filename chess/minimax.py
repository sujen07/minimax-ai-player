import chess
import random

def terminal(board):
    if board.is_checkmate() or board.is_stalemate():
        return True
    return False


CHECKMATE_SCORE = 100  # A large value for checkmate

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

def evaluate_move(board, move):
    """
    Evaluate the given move.
    This function can be as complex as needed, considering various chess strategies.
    """
    # Example: Prioritize captures
    if board.is_capture(move):
        return 10
    else:
        return 0

def sort_moves(board):
    legal_moves = list(board.legal_moves)
    scored_moves = [(move, evaluate_move(board, move)) for move in legal_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order
    sorted_moves = [move for move, score in scored_moves]
    return sorted_moves

def minimax(board, depth):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board=board):
        return None
    

    def optimize(board, depth=3, alpha=float('-inf'), beta=float('inf')):
        if terminal(board=board) or depth <= 0:
            return evaluate_board(board=board), None
        
        legal_moves = sort_moves(board)

        if board.turn == chess.WHITE:  # Maximizing player
            value = float('-inf')
            optimal_action = None

            for action in legal_moves:
                new_board = board.copy()
                new_board.push(action)
                score, _ = optimize(new_board, depth=depth -1, alpha=alpha, beta=beta)
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
                score, _ = optimize(new_board, depth=depth - 1, alpha=alpha, beta=beta)
                if score < value:
                    value = score
                    optimal_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, optimal_action

    
    return optimize(board, depth=depth)[1]
