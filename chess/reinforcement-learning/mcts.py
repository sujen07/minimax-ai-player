import numpy as np
from chess_environment import *

class Node:
    def __init__(self, board, parent=None, action=None, prior=0):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = {} # {action: child node}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def value_mean(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def select_child(self, c=1.41):
        # -child.value_mean(): a child's value is from its own mover's perspective,
        # which is this node's opponent, so it has to be negated before comparing.
        return max(self.children.values(), key=lambda child: -child.value_mean() + c * child.prior * np.sqrt(self.visits) / (1 + child.visits))

@torch.no_grad()
def expand_batch(model, nodes, device=None):
    """One forward pass shared by every node in `nodes`.

    Mutates each node's .children in place (one per legal move) and returns
    the leaf value for each node, in the same order. This is the batching
    lever: on a GPU/MPS backend a small network's single-position forward
    pass is dominated by launch overhead, not compute, so evaluating N leaves
    at once costs barely more than evaluating one.
    """
    x = torch.cat([board_to_tensor(node.board) for node in nodes])
    if device is not None:
        x = x.to(device)
    logits, values = model(x)

    for i, node in enumerate(nodes):
        masked_logits = masked_policy_logits(logits[i], node.board)
        probs = torch.softmax(masked_logits, dim=0)
        for move in node.board.legal_moves:
            prior = probs[move_to_index(move, node.board)].item()
            next_board = node.board.copy()
            next_board.push(move)
            node.children[move] = Node(next_board, parent=node, action=move, prior=prior)

    return values.tolist()


def expand(model, node, device=None):
    return expand_batch(model, [node], device=device)[0]


def mcts_search_batch(model, boards, num_simulations=100, c=1.41, device=None,
                       add_dirichlet_noise=False, dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """Run `num_simulations` of MCTS on len(boards) independent trees at once,
    batching every round's leaf evaluations into a single forward pass.

    This is root parallelization: statistics are NOT shared across trees, only
    the (expensive, GPU-bound) network calls are — each round, every tree
    contributes at most one leaf to a single batched expand_batch() call.
    """
    roots = [Node(b.copy()) for b in boards]
    expand_batch(model, roots, device=device)

    if add_dirichlet_noise:
        for root in roots:
            if root.children:
                noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
                for child, n in zip(root.children.values(), noise):
                    child.prior = (1 - dirichlet_eps) * child.prior + dirichlet_eps * n

    for _ in range(num_simulations):
        leaves = []
        for root in roots:
            node = root
            while not node.is_leaf():
                node = node.select_child(c)
            leaves.append(node)

        to_expand = []
        for node in leaves:
            if terminal(node.board):
                backpropagate(node, terminal_value(node.board))
            else:
                to_expand.append(node)

        if to_expand:
            values = expand_batch(model, to_expand, device=device)
            for node, value in zip(to_expand, values):
                backpropagate(node, value)

    return roots


def mcts_search(model, board, num_simulations=100, c=1.41, device=None, add_dirichlet_noise=False,
                dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """Single-tree MCTS; a thin convenience wrapper around mcts_search_batch."""
    return mcts_search_batch(
        model, [board], num_simulations=num_simulations, c=c, device=device,
        add_dirichlet_noise=add_dirichlet_noise,
        dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
    )[0]


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value  # flip for the opponent's perspective
        node = node.parent


def visit_distribution(root, temperature=1.0):
    """root's legal moves and their probabilities, derived from visit counts."""
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)

    if temperature == 0:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
    else:
        scaled = visits ** (1.0 / temperature)
        probs = scaled / scaled.sum()

    return moves, probs


def policy_target(root, temperature=1.0):
    """POLICY_SIZE-length training target: visit-count distribution over policy indices."""
    moves, probs = visit_distribution(root, temperature)
    target = torch.zeros(POLICY_SIZE)
    for move, p in zip(moves, probs):
        # += (not =) because under-promotions to different pieces share one index.
        target[move_to_index(move, root.board)] += p
    return target


def select_move(root, temperature=1.0):
    moves, probs = visit_distribution(root, temperature)
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]
