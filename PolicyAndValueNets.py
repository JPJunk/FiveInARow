import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Residual block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)  # ⚠️ Consider GroupNorm for small batches
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

# -----------------------------
# Policy network
# -----------------------------
class PolicyNet(nn.Module):
    """
    AlphaZero-style policy head.
    Input: [B, 2, board_size, board_size]
    Output: logits over board_size^2 intersections
    """
    def __init__(self, input_channels=2, board_size=15, num_blocks=7, channels=64):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size)
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return self.head(x)  # logits, apply softmax externally

# -----------------------------
# Value network
# -----------------------------
class ValueNet(nn.Module):
    """
    AlphaZero-style value head.
    Input: [B, 2, board_size, board_size]
    Output: scalar in [-1,1]
    """
    def __init__(self, input_channels=2, board_size=15, num_blocks=7, channels=64):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # clamp to [-1,1]
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        return self.head(x)  # shape [B,1]

# # class CompactPolicyCNN(nn.Module):
# #     """
# #     Compact CNN for Gomoku policy network.
# #     Input: [B, 2, 15, 15] (two channels: black stones, white stones)
# #     Output: logits over 225 board positions
# #     """
# #     def __init__(self, board_size=15):
# #         super().__init__()
# #         self.board_size = board_size
# #         flattened = board_size * board_size

# #         self.net = nn.Sequential(
# #             nn.Conv2d(2, 32, kernel_size=3, padding=1),  # preserve 15x15
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.Flatten(),
# #             nn.Linear(64 * board_size * board_size, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, flattened)  # logits for each intersection
# #         )

# #     def forward(self, x):
# #         return self.net(x)

# # class CompactValueCNN(nn.Module):
# #     """
# #     Compact CNN for Gomoku value network.
# #     Input: [B, 2, 15, 15]
# #     Output: scalar value in [-1,1]
# #     """
# #     def __init__(self, board_size=15):
# #         super().__init__()
# #         self.board_size = board_size

# #         self.net = nn.Sequential(
# #             nn.Conv2d(2, 32, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.Flatten(),
# #             nn.Linear(64 * board_size * board_size, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 1),
# #             nn.Tanh()  # clamp to [-1,1]
# #         )

# #     def forward(self, x):
# #         return self.net(x)
