import enum
import torch

from typing import Tuple

import level_test

# -----------------------------
# Constants and configs
# -----------------------------
BOARD_SIZE = 15
WIN_LENGTH = 5

# Encodings:
# 0 = empty, 1 = black (X), -1 = white (O)
BLACK = 1
WHITE = -1
# EMPTY = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Game enums & utility
# -----------------------------
# Training stages
stages = [
    level_test.TrainingStage(name="S0", sims_per_move=100, lr=1e-3, augmentation_phase=0),  # no aug, warmup
    level_test.TrainingStage(name="S1", sims_per_move=200, lr=5e-4, augmentation_phase=1),  # rotations
    level_test.TrainingStage(name="S2", sims_per_move=300, lr=5e-4, augmentation_phase=2),  # rotations + flip_ud
    level_test.TrainingStage(name="S3", sims_per_move=400, lr=3e-4, augmentation_phase=3),  # rotations + flip_lr
    level_test.TrainingStage(name="S4", sims_per_move=500, lr=3e-4, augmentation_phase=4),  # rotations + flip_ud + flip_lr
    level_test.TrainingStage(name="S5", sims_per_move=600, lr=3e-4, augmentation_phase=5),  # rotations + flip_ud + flip_lr + flip_ud_lr
]

# @dataclass
class Transition:
    def __init__(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done,
                 player,
                 q_value=0.0,
                 pi=None,
                 z=None,
                 meta=None):
        """
        Experience tuple for replay buffer.

        Args:
            state: board array before action
            action: integer index of move taken
            reward: scalar reward (post-game shaping)
            next_state: board array after action
            done: bool, whether terminal
            player: BLACK or WHITE who took the action
            q_value: optional Q estimate (from MCTS or value net)
            pi: optional MCTS visit distribution (numpy array length board_size^2)
            z: optional final outcome from this player's perspective (+1/-1/0)
            meta: optional dict for extra info
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.player = player
        self.q_value = q_value
        self.pi = pi
        self.z = z
        self.meta = meta or {}

class GameResult(enum.Enum):
    """Outcome of a finished game."""
    BLACK_WIN = enum.auto()
    WHITE_WIN = enum.auto()
    DRAW = enum.auto()
    ONGOING = enum.auto()

def index_to_coord(action: int, size: int = BOARD_SIZE) -> Tuple[int, int]:
    """Map flat action index to (row, col)."""
    return divmod(action, size)

def coord_to_index(row: int, col: int, size: int = BOARD_SIZE) -> int:
    """Map (row, col) to flat action index."""
    return row * size + col
