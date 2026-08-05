import tempfile
import unittest
from pathlib import Path

import chess
import torch

from chess_environment import (
    INPUT_CHANNELS,
    POLICY_SIZE,
    PositionRecord,
    SparsePolicy,
    board_to_tensor,
    index_to_move,
    move_to_index,
)
from mcts import (
    Node,
    mcts_search_batch,
    policy_target,
    training_repetition_exclusions,
)
from model import PolicyNetwork
from replay_buffer import ReplayBuffer
from train import load_checkpoint
from training_runtime import load_replay_buffer, save_replay_buffer


def repeated_board():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    return board


class ZeroModel:
    def __call__(self, x):
        return torch.zeros(x.shape[0], POLICY_SIZE), torch.zeros(x.shape[0])


class ChessRLV2Tests(unittest.TestCase):
    def test_history_and_rule_planes(self):
        board = repeated_board()
        record = PositionRecord.from_board(board)
        tensor = board_to_tensor(record)
        self.assertEqual(tensor.shape, (1, INPUT_CHANNELS, 8, 8))
        self.assertEqual(record.repetition_count, 2)
        self.assertTrue(torch.all(tensor[0, -1] == 0.5))

        no_castling = chess.Board()
        no_castling.castling_rights = chess.BB_EMPTY
        self.assertFalse(
            torch.equal(board_to_tensor(chess.Board()), board_to_tensor(no_castling))
        )

    def test_action_round_trip_including_promotions(self):
        boards = [
            chess.Board(),
            chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
            chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
        ]
        black_to_move = chess.Board()
        black_to_move.push_uci("e2e4")
        boards.append(black_to_move)
        for board in boards:
            indices = []
            for move in board.legal_moves:
                index = move_to_index(move, board)
                self.assertEqual(index_to_move(index, board), move)
                indices.append(index)
            self.assertEqual(len(indices), len(set(indices)))

    def test_repetition_history_survives_mcts_children(self):
        board = repeated_board()
        root = mcts_search_batch(ZeroModel(), [board], num_simulations=1)[0]
        self.assertTrue(root.children)
        child = root.children[chess.Move.from_uci("g1f3")]
        self.assertGreaterEqual(len(child.board.move_stack), len(board.move_stack))
        self.assertTrue(child.board.is_repetition(2))

    def test_training_target_excludes_immediate_third_repetition(self):
        board = chess.Board()
        for uci in (
            "g1f3", "g8f6", "f3g1", "f6g8",
            "g1f3", "g8f6", "f3g1",
        ):
            board.push_uci(uci)
        repeating_move = chess.Move.from_uci("f6g8")
        root = Node(board)
        root.visits = 10
        for move in board.legal_moves:
            child_board = board.copy()
            child_board.push(move)
            child = Node(child_board, parent=root, action=move, prior=0.1)
            child.visits = 1
            root.children[move] = child
        root.children[repeating_move].visits = 20

        excluded = training_repetition_exclusions(root)
        self.assertIn(repeating_move, excluded)
        target = policy_target(root, excluded_moves=excluded)
        self.assertNotIn(
            move_to_index(repeating_move, board), target.indices.tolist()
        )

        root.value_sum = -10
        self.assertEqual(training_repetition_exclusions(root), set())

    def test_training_filter_blocks_opponent_claim_on_next_move(self):
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            board.push_uci(uci)
        enabling_move = chess.Move.from_uci("f3g1")
        root = Node(board)
        root.visits = 10
        for move in board.legal_moves:
            child_board = board.copy()
            child_board.push(move)
            child = Node(child_board, parent=root, action=move, prior=0.1)
            child.visits = 1
            root.children[move] = child

        self.assertFalse(root.children[enabling_move].board.is_repetition(3))
        self.assertTrue(
            root.children[enabling_move].board.can_claim_threefold_repetition()
        )
        self.assertIn(enabling_move, training_repetition_exclusions(root))

    def test_sparse_loss_and_masked_value(self):
        board = chess.Board()
        move = next(iter(board.legal_moves))
        target = SparsePolicy(
            torch.tensor([move_to_index(move, board)]),
            torch.tensor([1.0]),
        )
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        total, policy_loss, value_loss = policy.alphazero_loss(
            [PositionRecord.from_board(board)],
            [target],
            torch.tensor([1.0]),
            value_weights=torch.tensor([0.0]),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(policy_loss))
        self.assertEqual(value_loss.item(), 0.0)

    def test_legacy_checkpoint_is_rejected(self):
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"policy_state_dict": policy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                load_checkpoint(path, policy)

    def test_replay_round_trip(self):
        board = chess.Board()
        move = next(iter(board.legal_moves))
        target = SparsePolicy(
            torch.tensor([move_to_index(move, board)]), torch.tensor([1.0])
        )
        source = ReplayBuffer(capacity=10)
        source.add([PositionRecord.from_board(board)], [target], [0.0], [1.0])
        restored = ReplayBuffer(capacity=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            save_replay_buffer(path, source)
            self.assertTrue(load_replay_buffer(path, restored))
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.sample(1)[0][0].current_fen, board.fen())


if __name__ == "__main__":
    unittest.main()
