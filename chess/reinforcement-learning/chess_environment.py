import chess
import torch

# Policy action space: from_square * 64 + to_square (queen promotions share the from-to index)
POLICY_SIZE = 64 * 64
CHECKMATE_SCORE = 40  # A large value for checkmate


def to_policy_square(square, turn):
    """
    Map a real-board square into the side-to-move policy frame.
    Black to move: vertical flip so the mover's pieces sit on the low ranks.
    """
    return chess.square_mirror(square) if turn == chess.BLACK else square


def board_to_tensor(board):
    """
    Encode the board from the side-to-move's perspective.
    Channels 0-5: current player's pieces (P,N,B,R,Q,K)
    Channels 6-11: opponent's pieces
    Spatially flipped when black to move so "my" pieces are always near rank 0.
    """
    tensor = torch.zeros(1, 12, 8, 8)
    turn = board.turn
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        channel = piece.piece_type - 1
        if piece.color != turn:
            channel += 6
        policy_sq = to_policy_square(square, turn)
        file_idx = chess.square_file(policy_sq)
        rank_idx = chess.square_rank(policy_sq)
        tensor[0, channel, file_idx, rank_idx] = 1
    return tensor


def outcome_reward(result):
    """Terminal reward from White's perspective (+1 win, -1 loss, 0 draw).

    Kept on the same [-1, 1] scale as evaluate_board / the shaping rewards so the
    value head isn't crushed by a scale mismatch between terminal and per-step signals.
    """
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def outcome_reward_for(result, turn):
    """outcome_reward converted to `turn`'s perspective (+1 turn wins, -1 turn loses)."""
    reward = outcome_reward(result)
    return reward if turn == chess.WHITE else -reward


def terminal_value(board):
    """Game value from the perspective of the side to move at this (terminal) board.

    A terminal board has no legal moves, so this is always <= 0: -1 if the side
    to move is checkmated, 0 for any other terminal state (stalemate, draw claims).
    """
    return -1.0 if board.is_checkmate() else 0.0


def tensor_to_board(tensor):
    board = chess.Board()
    for i in range(8):
        for j in range(8):
            piece = tensor[0, i, j]
            if piece != 0:
                board.set_piece_at(chess.square(i, j), piece)
    return board


def step(board, move):
    board.push(move)
    return board


def move_to_index(move, board):
    """Policy index for a real move, in the side-to-move (possibly flipped) frame."""
    turn = board.turn
    from_sq = to_policy_square(move.from_square, turn)
    to_sq = to_policy_square(move.to_square, turn)
    return from_sq * 64 + to_sq


def legal_move_mask(board, device=None, dtype=torch.float32):
    """Binary mask over the policy space: 1 = legal, 0 = illegal."""
    mask = torch.zeros(POLICY_SIZE, device=device, dtype=dtype)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = 1
    return mask


def masked_policy_logits(tensor, board):
    """Mask illegal move logits to -inf over the 64*64 policy space."""
    logits = tensor.reshape(-1)
    if logits.numel() < POLICY_SIZE:
        raise ValueError(f"Expected at least {POLICY_SIZE} logits, got {logits.numel()}")
    logits = logits[:POLICY_SIZE]

    mask = legal_move_mask(board, device=logits.device, dtype=logits.dtype)
    return logits.masked_fill(mask == 0, float("-inf"))


def index_to_move(index, board):
    """Map a policy-frame from*64+to index back to a legal chess.Move on the real board."""
    from_policy, to_policy = divmod(int(index), 64)
    turn = board.turn
    from_square = to_policy_square(from_policy, turn)
    to_square = to_policy_square(to_policy, turn)
    for move in board.legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            return move
    raise ValueError("No legal move found for index; board may have no legal moves")


def tensor_to_move(tensor, board):
    """Greedy legal move from policy logits (for evaluation / visualization)."""
    masked_logits = masked_policy_logits(tensor, board)
    return index_to_move(masked_logits.argmax().item(), board)


def legal_moves(board):
    return list(board.legal_moves)


def terminal(board):
    return board.is_game_over(claim_draw=True)


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
    num_pieces = len(board.piece_map())
    endgame = num_pieces < 14
    # Basic material count evaluation for non-terminal states
    score = 0
    for (piece, value) in [(chess.KNIGHT, 3), (chess.BISHOP, 3),
                           (chess.ROOK, 5), (chess.QUEEN, 9), (chess.PAWN, 1)]:
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value

    king_positions = {
        chess.WHITE: board.king(chess.WHITE),
        chess.BLACK: board.king(chess.BLACK)
    }
    
    for color, king_pos in king_positions.items():
        king_score = 0
        if endgame:
            # Higher score for king in corner or edge during endgame
            if king_pos in [0, 7, 56, 63]:  # Corners
                king_score = 3
            elif king_pos in [1, 6, 8, 15, 48, 55, 57, 62]:  # Edge but not corner
                king_score = 2
            else:
                king_score = 1
        else:
            # In middle game, central control might be more valuable
            # [Add logic for middle game king positioning if desired]
            pass

        if color == chess.WHITE:
            score += king_score
        else:
            score -= king_score

    # Normalize the score to be between -1 and 1
    # Note: This normalization is optional and depends on your specific use case
    normalized_score = score / CHECKMATE_SCORE
    return max(min(normalized_score, 1), -1)
