import random
from collections import deque


class ReplayBuffer:
    """Fixed-size FIFO buffer of (board, mcts_pi, outcome) self-play samples.

    Samples accumulate across self-play batches so gradient updates draw from
    a mix of recent and slightly-older games instead of only the batch that
    was just played, which smooths out the correlation between consecutive
    positions from the same game.
    """

    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.boards = deque(maxlen=capacity)
        self.pis = deque(maxlen=capacity)
        self.outcomes = deque(maxlen=capacity)

    def __len__(self):
        return len(self.boards)

    def add(self, boards, pis, outcomes):
        self.boards.extend(boards)
        self.pis.extend(pis)
        self.outcomes.extend(outcomes)

    def sample(self, batch_size):
        """Sample up to `batch_size` items uniformly at random, without replacement."""
        n = len(self.boards)
        batch_size = min(batch_size, n)
        idxs = random.sample(range(n), batch_size)
        boards = [self.boards[i] for i in idxs]
        pis = [self.pis[i] for i in idxs]
        outcomes = [self.outcomes[i] for i in idxs]
        return boards, pis, outcomes
