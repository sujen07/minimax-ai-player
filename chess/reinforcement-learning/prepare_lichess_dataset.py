#!/usr/bin/env python3
"""Convert Lichess open-database PGNs into Chess RL v2 replay shards.

Downloads (optional) monthly dumps from https://database.lichess.org/, streams
games, filters by rating / time control / result, and writes
ReplayBuffer-compatible .pt shards that drop straight into alphazero_loss.

Example:
  python prepare_lichess_dataset.py \\
    --month 2025-01 \\
    --min-elo 2000 \\
    --time-controls blitz,rapid,classical \\
    --max-games 50000 \\
    --out-dir data/lichess_sl
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import torch

from chess_environment import (
    ACTION_VERSION,
    ENCODING_VERSION,
    PositionRecord,
    SparsePolicy,
    move_to_index,
    outcome_reward_for,
)

LICHESS_DB_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)

# Approximate base seconds for Lichess TimeControl tags like "180+2".
TIME_CONTROL_BUCKETS = {
    "bullet": (0, 179),       # < 3 min
    "blitz": (180, 479),      # 3–8 min
    "rapid": (480, 1499),     # 8–25 min
    "classical": (1500, 10**9),
}

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
# Keep games that finished on the board or on the clock; drop abandons / rules issues.
ALLOWED_TERMINATIONS = {
    "Normal",
    "Time forfeit",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pgn",
        type=Path,
        help="Local .pgn or .pgn.zst file (or plain .zst of a PGN dump)",
    )
    source.add_argument(
        "--month",
        help="YYYY-MM dump to download from database.lichess.org into --download-dir",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/lichess_raw"),
        help="Where to store downloaded monthly dumps (default: data/lichess_raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lichess_sl"),
        help="Directory for replay .pt shards (default: data/lichess_sl)",
    )
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument(
        "--require-both-rated",
        action="store_true",
        help="Require both players to meet --min-elo (default: either side)",
    )
    parser.add_argument(
        "--time-controls",
        default="blitz,rapid,classical",
        help="Comma-separated buckets: bullet,blitz,rapid,classical (or 'all')",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many accepted games (default: no limit)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Stop after this many positions written (default: no limit)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Positions per output shard (default: 100000)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=500,
        help="Skip games longer than this many half-moves",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N scanned games",
    )
    return parser.parse_args()


def open_pgn_stream(path: Path):
    """Yield a text stream over a .pgn or zstd-compressed PGN dump."""
    path = Path(path)
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise SystemExit(
                "Reading .zst dumps requires the 'zstandard' package.\n"
                "  pip install zstandard"
            ) from exc
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return text, fh
    return path.open("r", encoding="utf-8", errors="replace"), None


def download_month(month: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = f"lichess_db_standard_rated_{month}.pgn.zst"
    dest = download_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached dump: {dest}")
        return dest

    url = LICHESS_DB_URL.format(month=month)
    print(f"Downloading {url}")
    print("Note: standard monthly dumps are large (often 15–30 GB compressed).")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"Saved to {dest}")
    return dest


def parse_elo(headers, key: str):
    raw = headers.get(key, "")
    if not raw or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def base_time_seconds(time_control: str):
    """Parse PGN TimeControl 'base+inc' → base seconds, or None if unknown."""
    if not time_control or time_control in {"-", "?"}:
        return None
    # Correspondence / unlimited
    if time_control == "0+0":
        return None
    base = time_control.split("+", 1)[0]
    try:
        return int(base)
    except ValueError:
        return None


def time_control_allowed(time_control: str, allowed_buckets: set[str] | None) -> bool:
    if allowed_buckets is None:
        return True
    base = base_time_seconds(time_control)
    if base is None:
        return False
    for name in allowed_buckets:
        lo, hi = TIME_CONTROL_BUCKETS[name]
        if lo <= base <= hi:
            return True
    return False


def game_passes_filters(game, args, allowed_buckets) -> bool:
    headers = game.headers
    result = headers.get("Result", "*")
    if result not in VALID_RESULTS:
        return False

    termination = headers.get("Termination", "Normal")
    if termination not in ALLOWED_TERMINATIONS:
        return False

    white_elo = parse_elo(headers, "WhiteElo")
    black_elo = parse_elo(headers, "BlackElo")
    if white_elo is None and black_elo is None:
        return False
    if args.require_both_rated:
        if white_elo is None or black_elo is None:
            return False
        if white_elo < args.min_elo or black_elo < args.min_elo:
            return False
    else:
        best = max(e for e in (white_elo, black_elo) if e is not None)
        if best < args.min_elo:
            return False

    if not time_control_allowed(headers.get("TimeControl", ""), allowed_buckets):
        return False

    return True


def game_to_samples(game):
    """Replay one game → lists of (PositionRecord, SparsePolicy, outcome, weight)."""
    result = game.headers["Result"]
    board = game.board()
    boards, pis, outcomes, weights = [], [], [], []

    for move_node in game.mainline():
        move = move_node.move
        if move not in board.legal_moves:
            return None
        try:
            index = move_to_index(move, board)
        except ValueError:
            return None

        boards.append(PositionRecord.from_board(board))
        pis.append(
            SparsePolicy(
                indices=torch.tensor([index], dtype=torch.int64),
                probabilities=torch.tensor([1.0], dtype=torch.float32),
            )
        )
        outcomes.append(outcome_reward_for(result, board.turn))
        weights.append(1.0)
        board.push(move)

        if len(boards) > 10_000:
            # Pathological PGN; bail.
            return None

    return boards, pis, outcomes, weights


def save_shard(out_dir: Path, shard_idx: int, boards, pis, outcomes, weights) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:05d}.pt"
    payload = {
        "encoding_version": ENCODING_VERSION,
        "action_version": ACTION_VERSION,
        "capacity": len(boards),
        "boards": boards,
        "pis": pis,
        "outcomes": outcomes,
        "value_weights": weights,
        "source": "lichess_supervised",
    }
    torch.save(payload, path)
    return path


def resolve_buckets(spec: str):
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return None
    buckets = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = buckets - set(TIME_CONTROL_BUCKETS)
    if unknown:
        raise SystemExit(f"Unknown time-control buckets: {sorted(unknown)}")
    return buckets


def main():
    args = parse_args()
    allowed_buckets = resolve_buckets(args.time_controls)

    if args.month:
        pgn_path = download_month(args.month, args.download_dir)
    else:
        pgn_path = args.pgn
        if not pgn_path.exists():
            raise SystemExit(f"PGN not found: {pgn_path}")

    stream, raw_handle = open_pgn_stream(pgn_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shard_boards, shard_pis, shard_outcomes, shard_weights = [], [], [], []
    shard_idx = 0
    scanned = accepted = skipped = 0
    positions = 0
    written_shards = []

    try:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            scanned += 1

            if scanned % args.progress_every == 0:
                print(
                    f"scanned={scanned:,} accepted={accepted:,} "
                    f"positions={positions:,} skipped={skipped:,}",
                    file=sys.stderr,
                )

            if not game_passes_filters(game, args, allowed_buckets):
                skipped += 1
                continue

            # Cheap ply-count gate before full replay: mainline length.
            ply_count = sum(1 for _ in game.mainline())
            if ply_count == 0 or ply_count > args.max_plies:
                skipped += 1
                continue

            samples = game_to_samples(game)
            if samples is None:
                skipped += 1
                continue

            boards, pis, outcomes, weights = samples
            accepted += 1
            positions += len(boards)

            shard_boards.extend(boards)
            shard_pis.extend(pis)
            shard_outcomes.extend(outcomes)
            shard_weights.extend(weights)

            while len(shard_boards) >= args.shard_size:
                chunk = args.shard_size
                path = save_shard(
                    args.out_dir,
                    shard_idx,
                    shard_boards[:chunk],
                    shard_pis[:chunk],
                    shard_outcomes[:chunk],
                    shard_weights[:chunk],
                )
                written_shards.append(path)
                print(f"wrote {path} ({chunk:,} positions)")
                shard_idx += 1
                del shard_boards[:chunk]
                del shard_pis[:chunk]
                del shard_outcomes[:chunk]
                del shard_weights[:chunk]

            if args.max_games is not None and accepted >= args.max_games:
                break
            if args.max_positions is not None and positions >= args.max_positions:
                break
    finally:
        stream.close()
        if raw_handle is not None:
            raw_handle.close()

    if shard_boards:
        path = save_shard(
            args.out_dir,
            shard_idx,
            shard_boards,
            shard_pis,
            shard_outcomes,
            shard_weights,
        )
        written_shards.append(path)
        print(f"wrote {path} ({len(shard_boards):,} positions)")

    manifest = {
        "encoding_version": ENCODING_VERSION,
        "action_version": ACTION_VERSION,
        "source_pgn": str(pgn_path),
        "min_elo": args.min_elo,
        "require_both_rated": args.require_both_rated,
        "time_controls": args.time_controls,
        "games_scanned": scanned,
        "games_accepted": accepted,
        "games_skipped": skipped,
        "positions": positions,
        "shards": [str(p) for p in written_shards],
    }
    manifest_path = args.out_dir / "manifest.pt"
    torch.save(manifest, manifest_path)

    print()
    print(f"Done. accepted_games={accepted:,} positions={positions:,}")
    print(f"Shards: {len(written_shards)} under {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(
        "Next: train a supervised warm-start on these shards, then resume "
        "self-play with multiprocess_train.py."
    )


if __name__ == "__main__":
    main()
