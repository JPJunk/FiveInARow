import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

import symmetry_utils

from Gomoku import Gomoku
from Utils import Transition, GameResult, BLACK, WHITE, DEVICE
from PolicyAndValueNets import PolicyNet, ValueNet
from ReplayBuffer import ReplayBuffer

import os
os.chdir(r"C:\Users\...\Documents\PythonProjects\Gomoku DRL")

# -----------------------------
# DRL Agent (PyTorch)
# -----------------------------

class DRL:
    """
    Deep RL agent for Gomoku using ResNet-style policy and value networks.
    Includes epsilon-greedy action selection and reward shaping.
    """
    def __init__(self, board_size=15,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Networks
        # self.policy = PolicyNet(board_size=board_size).to(self.device)
        # self.value  = ValueNet(board_size=board_size).to(self.device)

        # policy_net 
        self.policy = PolicyNet(board_size=15)
        bias = gomoku_centrality_bias_for_policy(board_size=self.policy.board_size, scale=0.1) # 0.1
        with torch.no_grad():
            # head[-1] is the final Linear layer
            self.policy.head[-1].bias.copy_(bias)

        # value_net
        self.value = ValueNet(board_size=15)
        bias = gomoku_centrality_bias_for_value(board_size=self.value.board_size, scale=0.05) # 0.05
        with torch.no_grad():
            # head[3] is the Linear(board_size*board_size -> 64)
            self.value.head[4].bias.copy_(bias[:self.value.board_size**2].mean().expand(64))

        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=5e-4, weight_decay=1e-4)  # ORIGINAL 1e-3
        self.value_opt  = optim.Adam(self.value.parameters(), lr=5e-4, weight_decay=1e-4) # ORIGINAL 1e-3

        # Replay buffer, epsilon schedule
        self.replay = ReplayBuffer(capacity=40000) # ORIGINAL 10000
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        # self.epsilon_decay_steps = 10000 # TODO: SHOULD THIS BE RAISED? (WHEN REPLAY BUFFER IS RAISED)
        self.epsilon_decay_steps = 40000  # ORIGINAL 10000
        self.epsilon_step = 0
        self.temperature = 1.0

        # Discount factor
        self.gamma = 0.99

        # Game counter for logging
        self.game_counter = 0

        self.board_size = board_size


        #self.augmentation_phase = 0

    # --------- Push transition with Symmetry augmentation ----------
    def store_transition(self, t):
        # Always store the original
        self.replay.push(t)

        # Phase 1: rotations only
        if 1000 <= self.game_counter < 2000: # 40k
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))

        # Phase 2: rotations + flips (u/d)                
        if 2000 <= self.game_counter < 3000: # 80k
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))


        # Phase 3: rotations + flips (l/r)
        if 3000 <= self.game_counter < 4000: # 80k
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))

        # Phase 4: rotations + both flips
        if 4000 <= self.game_counter < 5000: # 160k
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))

        # Phase 5: all symmetries
        if self.game_counter >= 5000: # 320k
            for k in (1, 2, 3):
                self.replay.push(symmetry_utils.rotate_transition(t, self.board_size, k))
            self.replay.push(symmetry_utils.flip_ud_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_lr_transition(t, self.board_size))
            self.replay.push(symmetry_utils.flip_ud_lr_transition(t, self.board_size))

    def evolve_transitions(self, num_parents=2, num_offspring=100, q_threshold=0.6):
        self.replay.evolve_transitions(num_parents, num_offspring, q_threshold) 

    # --------- State encoding ----------
    def encode_state(self, board, current_player):
        """Convert board to tensor with perspective of current_player."""
        if board is None:
            arr = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        else:
            arr = np.array(board, dtype=np.int8, copy=False)
        black = (arr == BLACK).astype(np.float32)
        white = (arr == WHITE).astype(np.float32)
        if current_player == BLACK:
            x = np.stack([black, white], axis=0)
        else:
            x = np.stack([white, black], axis=0)
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    # --------- Epsilon decay ----------
    def _decay_epsilon(self):
        """Linear epsilon decay."""
        if self.epsilon_step < self.epsilon_decay_steps:
            delta = (self.epsilon - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_end, self.epsilon - delta)
            self.epsilon_step += 1

    # --------- Action selection ----------
    def select_action(self, env: Gomoku, legal_moves: list[int]) -> int:
        """
        Epsilon-greedy action using policy logits.
        For NN vs NN, set epsilon low; for training, start high and decay.
        """
        self._decay_epsilon()
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        state_t = self.encode_state(env.board, env.current_player)
        logits = self.policy(state_t).squeeze(0)  # [225]

        # Mask illegal moves by setting logits to very negative
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_moves] = 0.0
        masked_logits = logits + mask

        # Sample from softmax distribution
        probs = torch.softmax(masked_logits / max(self.temperature, 1e-6), dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        return action_idx

    # --------- Training hook ----------
    def train_after_game(self, batch_size: int = 256,
                        policy_coeff: float = 1.0,
                        value_coeff: float = 1.0) -> dict:
        """
        Perform backprop after each completed game using replay samples.
        Always returns a stats dict with keys: policy_loss, value_loss, total_loss.
        """

        # Default stats if not enough data
        if len(self.replay) < 10:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": 0.0,
            }

        batch = self.replay.sample(batch_size)

        # Prepare tensors
        states, actions, rewards, next_states, dones, players = [], [], [], [], [], []
        for tr in batch:
            states.append(self.encode_state(tr.state, tr.player))
            actions.append(tr.action)
            rewards.append(tr.reward)
            next_states.append(self.encode_state(tr.next_state, tr.player))
            dones.append(tr.done)
            players.append(tr.player)

        states_t = torch.cat(states, dim=0)       # [B, 2, H, W]
        next_states_t = torch.cat(next_states, 0) # [B, 2, H, W]
        actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        # Policy loss (stub): cross-entropy toward actions with advantages
        logits = self.policy(states_t)  # [B, 225]
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions_t.view(-1, 1)).squeeze(1)

        # Value targets using 1-step TD
        with torch.no_grad():
            next_values = self.value(next_states_t).squeeze(1)  # [B]
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_values

        values = self.value(states_t).squeeze(1)  # [B]
        advantages = (targets - values).detach()

        # Losses
        policy_loss = -(advantages * chosen_log_probs).mean()
        value_loss = nn.MSELoss()(values, targets)
        total_loss = policy_coeff * policy_loss + value_coeff * value_loss

        # Optimize
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), max_norm=1.0)
        self.policy_opt.step()
        self.value_opt.step()

        # --- Logging ---
        self.game_counter = getattr(self, "game_counter", 0)
        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }
        print(f"[Train] Game {self.game_counter} stats: {stats}")

        # timestamp = datetime.now().isoformat(timespec="seconds")
        # with open("train_after_game.txt", "a") as f:
        #     f.write(f"Game {self.game_counter} [{timestamp}]: "
        #             f"policy_loss={stats['policy_loss']:.4f}, "
        #             f"value_loss={stats['value_loss']:.4f}, "
        #             f"total_loss={stats['total_loss']:.4f}\n")

        return stats
    
    # --------- AlphaZero-style training ----------
    def train_after_game_az(self, batch_size: int = 256,
                        policy_coeff: float = 1.0,
                        value_coeff: float = 1.0) -> dict:
        """AlphaZero-style training: policy head imitates MCTS π, value head predicts final outcome z."""

        if len(self.replay) < 10:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        # batch = self.replay.sample(batch_size)
        batch = [tr for tr in self.replay.sample(batch_size) if tr.pi is not None and tr.z is not None]

        # Prepare tensors
        states = [self.encode_state(tr.state, tr.player) for tr in batch]
        states_t = torch.cat(states, dim=0)  # [B, 2, H, W]

        logits = self.policy(states_t)  # [B, board_size^2]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Policy target π (visit distribution)
        pi_t = torch.tensor([tr.pi for tr in batch],
                            dtype=torch.float32, device=self.device)
        policy_loss = -(pi_t * log_probs).sum(dim=1).mean()

        # Value target z (final outcome from perspective of tr.player)
        z_t = torch.tensor([tr.z for tr in batch],
                        dtype=torch.float32, device=self.device)
        values = self.value(states_t).squeeze(1)  # [B]
        value_loss = nn.MSELoss()(values, z_t)

        total_loss = policy_coeff * policy_loss + value_coeff * value_loss

        # Optimize
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.parameters()) +
                                list(self.value.parameters()), max_norm=1.0)
        self.policy_opt.step()
        self.value_opt.step()

        # Logging
        self.game_counter = getattr(self, "game_counter", 0)
        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item()),
        }
        print(f"[Train] Game {self.game_counter} stats: {stats}")

        # timestamp = datetime.now().isoformat(timespec="seconds")
        # with open("train_after_game.txt", "a") as f:
        #     f.write(f"Game {self.game_counter} [{timestamp}]: "
        #             f"policy_loss={stats['policy_loss']:.4f}, "
        #             f"value_loss={stats['value_loss']:.4f}, "
        #             f"total_loss={stats['total_loss']:.4f}\n")

        return stats

    # --------- Reward shaping ----------
    @staticmethod
    def compute_rewards(result: GameResult, moves: list[Transition]) -> None:
        """
        Assign rewards to transitions post-game.
        - Winner gets +1 on its moves, loser gets -1.
        - Draw = 0 for all moves.
        """
        if result == GameResult.DRAW:
            for t in moves:
                t.reward = 0.5 # for learning defence # original 0.0
            return

        winner = BLACK if result == GameResult.BLACK_WIN else WHITE
        for t in moves:
            t.reward = 1.0 if t.player == winner else -2.0 # for learning offence # original -1.0



def gomoku_centrality_bias_for_policy(board_size=15, scale=0.1):
    """
    Compute a centrality heatmap: number of 5-in-a-row lines each cell participates in.
    Returns a flattened bias vector scaled to ~scale.
    """
    heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    win_len = 5

    for r in range(board_size):
        for c in range(board_size):
            count = 0
            # horizontal
            for cc in range(c - win_len + 1, c + 1):
                if 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1
            # vertical
            for rr in range(r - win_len + 1, r + 1):
                if 0 <= rr and rr + win_len - 1 < board_size:
                    count += 1
            # diag \
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c + d
                if 0 <= rr and rr + win_len - 1 < board_size and 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1
            # diag /
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c - d
                if 0 <= rr and rr + win_len - 1 < board_size and 0 <= cc - (win_len - 1) and cc < board_size:
                    count += 1
            heatmap[r, c] = count * count * count

    bias = heatmap.flatten()
    bias = bias / bias.max() * scale  # normalize and scale
    return torch.tensor(bias, dtype=torch.float32)

def gomoku_centrality_bias_for_value(board_size=15, scale=0.1):
    """
    Compute a centrality heatmap: number of 5-in-a-row lines each cell participates in.
    Returns a flattened bias vector scaled to ~scale.
    """
    heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    win_len = 5

    for r in range(board_size):
        for c in range(board_size):
            count = 0
            # horizontal
            for cc in range(c - win_len + 1, c + 1):
                if 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1
            # vertical
            for rr in range(r - win_len + 1, r + 1):
                if 0 <= rr and rr + win_len - 1 < board_size:
                    count += 1
            # diag \
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c + d
                if 0 <= rr and rr + win_len - 1 < board_size and 0 <= cc and cc + win_len - 1 < board_size:
                    count += 1
            # diag /
            for d in range(-win_len + 1, 1):
                rr, cc = r + d, c - d
                if 0 <= rr and rr + win_len - 1 < board_size and 0 <= cc - (win_len - 1) and cc < board_size:
                    count += 1
            heatmap[r, c] = count

    bias = heatmap.flatten()
    bias = bias / bias.max() * scale  # normalize and scale
    return torch.tensor(bias, dtype=torch.float32)
