import sys
import time
from collections import Counter, deque
from pathlib import Path

import chess
import torch
from model import PolicyNetwork
from chess_environment import *
from mcts import mcts_search_batch, policy_target, select_move
from replay_buffer import ReplayBuffer

CHESS_DIR = Path(__file__).resolve().parent.parent
RL_DIR = Path(__file__).resolve().parent

# The pygame UI is only needed when visualizing. Import it lazily so headless
# training (e.g. on a server with no display) never depends on pygame.
chess_ui = None


def _ensure_ui():
    """Import chess/main.py (pygame UI) on demand. Returns the module."""
    global chess_ui
    if chess_ui is None:
        if str(CHESS_DIR) not in sys.path:
            sys.path.insert(0, str(CHESS_DIR))
        import main as _chess_ui  # noqa: WPS433 (intentional lazy import)

        chess_ui = _chess_ui
    return chess_ui


def _pump_events():
    """Process pygame events; return False if the user closed the window."""
    ui = _ensure_ui()
    for event in ui.pygame.event.get():
        if event.type == ui.pygame.QUIT:
            return False
    return True


def _draw_position(board, last_move=None, caption=None):
    ui = _ensure_ui()
    if caption:
        ui.pygame.display.set_caption(caption)
    ui.draw_board_and_pieces(
        ui.screen,
        board,
        ui.pieces,
        selected_piece=None,
        blink_timer=1,
        last_ai_move=last_move,
        possible_moves=[],
        player=chess.WHITE,
    )
    ui.pygame.display.flip()


def save_checkpoint(path, policy, optimizer, games_done, results_window):
    """Persist full training state so a run can be resumed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "games_done": games_done,
            "results_window": list(results_window),
        },
        path,
    )


def load_checkpoint(path, policy, optimizer=None, device=None):
    """Restore training state saved by save_checkpoint. Returns a metadata dict."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Ensure any optimizer state tensors (exp_avg, etc.) live on `device`,
        # otherwise a resumed step mixes CPU/MPS tensors and crashes.
        for state in optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device)
    return {
        "games_done": ckpt.get("games_done", 0),
        "results_window": ckpt.get("results_window", []),
    }


def self_play_games_batch(policy, device, num_games, num_simulations, c_puct, temp_threshold,
                           dirichlet_alpha, dirichlet_eps, max_moves=200,
                           on_move=None, vis_index=None):
    """Play `num_games` self-play games concurrently (root parallelization).

    Every ply, all still-running games advance one MCTS search together, and
    mcts_search_batch shares their leaf evaluations across a single forward
    pass per simulation round instead of one pass per game — the lever that
    actually matters on a GPU, since a small network's single-position forward
    pass is dominated by launch overhead rather than compute.

    Returns (boards, pis, outcomes, results, terminations): the first three
    are training samples flattened across every game; `results` and
    `terminations` are one entry per game (len == num_games). `terminations`
    names why each game ended (e.g. "CHECKMATE", "THREEFOLD_REPETITION",
    "MOVE_CAP" for hitting the ply cap without a ruled game-over, or
    "INTERRUPTED" if the visualization window was closed mid-game).
    on_move(board, move), if given, is called after `vis_index`'s game makes
    a move and should return False to stop every game immediately.
    """
    boards = [chess.Board() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]  # per game: (board_before, pi, mover)
    plies = [0] * num_games
    done = [False] * num_games
    results = [None] * num_games
    terminations = [None] * num_games

    while not all(done):
        active = [i for i in range(num_games) if not done[i]]
        roots = mcts_search_batch(
            policy, [boards[i] for i in active],
            num_simulations=num_simulations, c=c_puct, device=device,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
        )

        for root, i in zip(roots, active):
            temperature = 1.0 if plies[i] < temp_threshold else 0.0
            # Store the temperature=1 (raw visit-count) distribution as the
            # training target regardless of temperature used to pick the move.
            pi = policy_target(root, temperature=1.0)
            move = select_move(root, temperature=temperature)

            histories[i].append((boards[i].copy(), pi, boards[i].turn))
            boards[i].push(move)
            plies[i] += 1

            game_over = boards[i].is_game_over(claim_draw=True)
            if game_over or plies[i] >= max_moves:
                done[i] = True
                if game_over:
                    results[i] = boards[i].result(claim_draw=True)
                    terminations[i] = boards[i].outcome(claim_draw=True).termination.name
                else:
                    results[i] = "1/2-1/2"
                    terminations[i] = "MOVE_CAP"

            if i == vis_index and on_move is not None and not on_move(boards[i], move):
                done = [True] * num_games
                break

    boards_flat, pis_flat, outcomes_flat = [], [], []
    for i in range(num_games):
        # Move-cap or user-interrupted games are treated as drawn (no signal to learn).
        result = results[i] if results[i] is not None else "1/2-1/2"
        terminations[i] = terminations[i] or "INTERRUPTED"
        for board_before, pi, mover in histories[i]:
            boards_flat.append(board_before)
            pis_flat.append(pi)
            outcomes_flat.append(outcome_reward_for(result, mover))

    return boards_flat, pis_flat, outcomes_flat, results, terminations


def train(
    policy,
    optimizer,
    episodes,
    device,
    games_per_batch=4,
    num_simulations=100,
    c_puct=1.41,
    temp_threshold=15,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    value_coef=1.0,
    epochs_per_batch=2,
    max_moves=200,
    visualize=False,
    move_delay=0.15,
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    start_games=0,
    results_window=None,
    buffer_capacity=50_000,
    train_batch_size=2048,
    min_buffer_size=1024,
):
    """
    Train the policy AlphaZero-style: play `games_per_batch` self-play games
    concurrently (root-parallel MCTS — see mcts_search_batch in mcts.py, which
    batches every simulation round's leaf evaluations across all of them into
    one forward pass), push the resulting (position, MCTS visit-count policy,
    game outcome) samples into a replay buffer, then fit the network against
    minibatches sampled from that buffer via alphazero_loss.

    games_per_batch: self-play games run concurrently before each update
    num_simulations: MCTS simulations run per move
    c_puct: PUCT exploration constant used by MCTS's child selection
    temp_threshold: plies per game before move sampling switches from the
        visit-count distribution (temperature=1, exploration) to greedy
        (temperature=0, exploitation); the stored training target always
        uses the temperature=1 distribution regardless of the move actually played
    dirichlet_alpha / dirichlet_eps: root exploration noise added during self-play
    value_coef: weight on the value-head MSE term in alphazero_loss
    epochs_per_batch: number of gradient-update passes per self-play batch, each
        over a fresh minibatch sampled from the replay buffer
    max_moves: ply cap per self-play game (treated as a draw if reached)
    visualize: render the first game of each batch in the chess UI
    move_delay: seconds to pause after each rendered move (visualized game only)
    checkpoint_dir: directory for periodic checkpoints + latest.pt
    checkpoint_every: save a checkpoint every N batches
    start_games / results_window: for resuming a previous run
    buffer_capacity: max samples kept in the replay buffer (oldest evicted first)
    train_batch_size: samples drawn from the buffer per gradient update
    min_buffer_size: skip training until the buffer holds at least this many
        samples, so early updates aren't fit on a handful of positions
    """
    policy.to(device)
    running = True

    if results_window is None:
        results_window = deque(maxlen=100)
    else:
        results_window = deque(results_window, maxlen=100)
    draw_reason_window = deque(maxlen=100)

    ckpt_dir = Path(checkpoint_dir)
    games_done = start_games
    batch_idx = 0
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    run_start = time.time()

    while games_done < episodes and running:
        batch_start = time.time()
        batch_size = min(games_per_batch, episodes - games_done)

        show = visualize and batch_size > 0
        if show:
            _draw_position(
                chess.Board(),
                caption=(
                    f"RL Chess — batch {batch_idx + 1} ({batch_size} games), "
                    f"{games_done}/{episodes} games done"
                ),
            )

        def on_move(board, move):
            nonlocal running
            running = _pump_events()
            if running:
                _draw_position(board, last_move=move)
                time.sleep(move_delay)
            return running

        all_boards, all_pis, all_outcomes, batch_results, batch_terminations = self_play_games_batch(
            policy, device,
            num_games=batch_size,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temp_threshold=temp_threshold,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_moves=max_moves,
            on_move=on_move if show else None,
            vis_index=0 if show else None,
        )

        if not running:
            break

        games_done += batch_size
        batch_idx += 1

        if all_boards:
            replay_buffer.add(all_boards, all_pis, all_outcomes)

            # Each epoch draws a fresh minibatch from the replay buffer, so
            # updates see a mix of this batch and earlier games rather than
            # repeatedly fitting the same freshly-played positions.
            loss = None
            if len(replay_buffer) >= min_buffer_size:
                for _ in range(epochs_per_batch):
                    sample_boards, sample_pis, sample_outcomes = replay_buffer.sample(
                        train_batch_size
                    )
                    outcomes_tensor = torch.tensor(sample_outcomes, dtype=torch.float32)

                    optimizer.zero_grad()
                    loss = policy.alphazero_loss(
                        sample_boards, sample_pis, outcomes_tensor, value_coef=value_coef,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()

            results_window.extend(batch_results)
            wins = results_window.count("1-0")
            losses = results_window.count("0-1")
            draws = results_window.count("1/2-1/2")
            n = max(len(results_window), 1)

            draw_reason_window.extend(
                termination for result, termination in zip(batch_results, batch_terminations)
                if result == "1/2-1/2"
            )
            draw_reason_counts = Counter(draw_reason_window)
            draw_reason_str = ", ".join(
                f"{reason}={count}" for reason, count in draw_reason_counts.most_common()
            ) or "n/a"

            loss_str = f"{loss.item():.4f}" if loss is not None else "n/a (filling buffer)"
            batch_elapsed = time.time() - batch_start
            run_elapsed = time.time() - run_start
            print(
                f"[{time.strftime('%H:%M:%S')} | batch {batch_elapsed:.1f}s | "
                f"total {run_elapsed / 60:.1f}m] "
                f"Batch {batch_idx} | games {games_done}/{episodes} | "
                f"loss={loss_str} | "
                f"buffer={len(replay_buffer)}/{buffer_capacity} | "
                f"[last {n}: W{wins}/D{draws}/L{losses} "
                f"white_win%={100 * wins / n:.0f}] | "
                f"draws[last {len(draw_reason_window)}]: {draw_reason_str}"
            )

        # Periodic checkpointing.
        if checkpoint_every and batch_idx % checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"ckpt_{games_done:06d}.pt",
                policy,
                optimizer,
                games_done,
                results_window,
            )
            save_checkpoint(
                ckpt_dir / "latest.pt",
                policy,
                optimizer,
                games_done,
                results_window,
            )
            print(f"  ↳ checkpoint saved at {games_done} games")

    # Always save a final checkpoint on exit (also covers early window-close).
    save_checkpoint(
        ckpt_dir / "latest.pt",
        policy,
        optimizer,
        games_done,
        results_window,
    )

    if visualize and chess_ui is not None:
        chess_ui.pygame.quit()

    return policy


def visualize_training(policy, device, games=3, move_delay=0.4):
    """Play self-play games with the policy and render them using chess/main.py UI."""
    policy.eval()
    pygame = _ensure_ui().pygame
    pygame.display.set_caption("RL Chess — Training Visualization")

    running = True
    for game_idx in range(games):
        if not running:
            break

        board = chess.Board()
        last_move = None
        _draw_position(board)

        while not board.is_game_over(claim_draw=True) and running:
            running = _pump_events()
            if not running:
                break

            x = board_to_tensor(board).to(device)
            logits, _ = policy(x)
            move = tensor_to_move(logits, board)

            board.push(move)
            last_move = move
            _draw_position(board, last_move=last_move)
            time.sleep(move_delay)

        if running:
            print(f"Game {game_idx + 1}/{games} result: {board.result(claim_draw=True)}")
            end_pause = time.time() + 1.5
            while time.time() < end_pause and running:
                running = _pump_events()
                pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the RL chess policy.")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="total number of self-play games")
    parser.add_argument("--games-per-batch", type=int, default=4,
                        help="self-play games run concurrently (batched MCTS) before each update")
    parser.add_argument("--num-simulations", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.41,
                        help="PUCT exploration constant")
    parser.add_argument("--temp-threshold", type=int, default=15,
                        help="plies before move selection switches from sampling to greedy")
    parser.add_argument("--epochs-per-batch", type=int, default=2,
                        help="gradient-update passes per self-play batch, each over a "
                             "fresh minibatch sampled from the replay buffer")
    parser.add_argument("--value-coef", type=float, default=1.0,
                        help="weight on the value-head MSE loss term")
    parser.add_argument("--buffer-capacity", type=int, default=50_000,
                        help="max samples kept in the replay buffer")
    parser.add_argument("--train-batch-size", type=int, default=2048,
                        help="samples drawn from the replay buffer per gradient update")
    parser.add_argument("--min-buffer-size", type=int, default=1024,
                        help="skip training until the buffer holds at least this many samples")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="ply cap per self-play game (treated as a draw if reached)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Adam learning rate")
    parser.add_argument("--no-visualize", action="store_true",
                        help="train headless (no pygame window)")
    parser.add_argument("--move-delay", type=float, default=0.1,
                        help="seconds between rendered moves of the shown game")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(RL_DIR / "checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save a checkpoint every N batches")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a checkpoint to resume from")
    parser.add_argument("--model-out", type=str,
                        default=str(RL_DIR / "chess_rl_model.pt"),
                        help="where to save the final trained weights")
    args = parser.parse_args()

    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    start_games = 0
    results_window = None
    if args.resume:
        meta = load_checkpoint(args.resume, policy, optimizer, device=device)
        start_games = meta["games_done"]
        results_window = meta["results_window"]
        print(f"Resumed from {args.resume} at {start_games} games")

    policy = train(
        policy,
        optimizer,
        episodes=args.episodes,
        device=device,
        games_per_batch=args.games_per_batch,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temp_threshold=args.temp_threshold,
        epochs_per_batch=args.epochs_per_batch,
        value_coef=args.value_coef,
        max_moves=args.max_moves,
        buffer_capacity=args.buffer_capacity,
        train_batch_size=args.train_batch_size,
        min_buffer_size=args.min_buffer_size,
        visualize=not args.no_visualize,
        move_delay=args.move_delay,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        start_games=start_games,
        results_window=results_window,
    )

    # Save just the final model weights (lighter than a full checkpoint).
    torch.save(policy.state_dict(), args.model_out)
    print(f"Final model weights saved to {args.model_out}")
