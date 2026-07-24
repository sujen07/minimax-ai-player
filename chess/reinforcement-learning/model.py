import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_environment import board_to_tensor, legal_move_mask, move_to_index, POLICY_SIZE


def _group_norm(channels, max_groups=16):
    """GroupNorm with a group count that divides `channels`.

    GroupNorm (unlike BatchNorm) is independent of batch size, so it stays valid
    when self-play rollouts shrink the batch down to a single active game.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Standard pre-activation-free residual block: (conv-norm-relu) x2 + skip."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class PolicyNetwork(nn.Module):
    """Residual CNN with separate policy and value heads (AlphaZero-style trunk).

    channels / num_blocks control capacity. Defaults are a deeper, wider network
    than the original 3-conv stack while keeping the same (logits, value) API.
    """

    def __init__(self, channels=64, num_blocks=6, policy_head_channels=32,
                 value_head_channels=8, value_hidden=256):
        super().__init__()

        # Input stem: lift the 12 piece planes into the trunk width.
        self.stem = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk.
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head: 3x3 conv -> flatten -> linear over the 64*64 move space.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_head_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(policy_head_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_head_channels * 8 * 8, POLICY_SIZE)

        # Value head: 1x1 conv -> MLP -> scalar in [-1, 1] (matches reward scale).
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_head_channels, kernel_size=1, bias=False),
            _group_norm(value_head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_head_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        p = self.policy_conv(x)
        logits = self.policy_fc(p.flatten(1))

        v = self.value_conv(x)
        value = torch.tanh(self.value_fc(v.flatten(1))).squeeze(-1)
        return logits, value

    def alphazero_loss(self, boards, mcts_policies, outcomes, value_coef=1.0):
        device = next(self.parameters()).device

        x = torch.cat([board_to_tensor(board) for board in boards]).to(device)
        logits_batch, values = self.forward(x)
        logits_batch = logits_batch[:, :POLICY_SIZE]

        policy_losses = []
        for i, board in enumerate(boards):
            mask = legal_move_mask(board, device=logits_batch.device, dtype=logits_batch.dtype)
            masked_logits = logits_batch[i].masked_fill(mask == 0, float("-inf"))
            log_probs_i = F.log_softmax(masked_logits, dim=0)
            # mcts_policies[i] is a target distribution over the same move-index space,
            # zero everywhere except the moves MCTS actually visited
            target = mcts_policies[i].to(device=device, dtype=log_probs_i.dtype)
            # cross-entropy against a soft target distribution, safely handling -inf
            safe_log_probs_i = log_probs_i.masked_fill(mask == 0, 0.0)
            policy_losses.append(-(target * safe_log_probs_i).sum())

        policy_loss = torch.stack(policy_losses).mean()
        values = values.reshape(-1)
        outcomes = outcomes.to(device=device, dtype=values.dtype)
        value_loss = F.mse_loss(values, outcomes)

        return policy_loss + value_coef * value_loss

        