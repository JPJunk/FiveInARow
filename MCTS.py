import math
import numpy as np
import torch
import torch.nn.functional as F

# from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from collections import defaultdict
from datetime import datetime

from Utils import GameResult, BLACK, WHITE, BOARD_SIZE, DEVICE
from Gomoku import Gomoku

import os
os.chdir(r"C:\Users\...\Documents\PythonProjects\Gomoku DRL")

# -----------------------------
# MCTS Node
# -----------------------------
class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'] = None,
                 action_from_parent: Optional[int] = None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.P: Dict[int, float] = {}              # priors
        self.N: Dict[int, int] = defaultdict(int)  # visit counts
        self.W: Dict[int, float] = defaultdict(float)  # total value
        self.Q: Dict[int, float] = defaultdict(float)  # mean value
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded: bool = False
        self.current_player: Optional[int] = None
        self.value_eval: Optional[float] = None    # <-- add this

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25):
        """Inject Dirichlet noise into root priors to encourage exploration."""
        if not self.is_expanded or not self.P:
            return
        actions = list(self.P.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.P[a] = (1 - epsilon) * self.P[a] + epsilon * float(n)

# -----------------------------
# MCTS Core
# -----------------------------
from MCTS import MCTSNode  # assumes your MCTSNode as defined earlier

class MCTS:
    """
    AlphaZero-style MCTS.
    - Uses policy/value nets to expand/evaluate
    - Builds π from root visit counts
    - Robust to np.int64 vs int key mismatches
    """
    def __init__(
        self,
        env_cls,
        encode_state_fn,
        policy_net,
        value_net,
        sims_per_move: int = 200,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.env_cls = env_cls
        self.encode_state = encode_state_fn
        self.policy_net = policy_net
        self.value_net = value_net
        self.sims_per_move = sims_per_move
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.average_q_at_root = 0.0
        self.limit_rollout_depth = 5  # max depth before cutoff

    def _focused_legal(self, env: Gomoku, radius: int = 2):
        legal = env.get_valid_moves()
        if env.moves_played == 0:
            return [int(a) for a in legal]  # full set for first move
        stones = np.argwhere(env.board != 0)
        if stones.size == 0:
            return [int(a) for a in legal]
        allowed = []
        size = env.size
        for a in legal:
            r, c = divmod(int(a), size)
            dmin = np.min(np.abs(stones[:,0] - r) + np.abs(stones[:,1] - c))
            if dmin <= radius:
                allowed.append(int(a))
        return allowed if allowed else [int(a) for a in legal]

    # Selection: return path of (node, action) edges and final env
    def _select(self, node: MCTSNode, env: Gomoku):
        path = []  # list of (node, action)
        if not node.is_expanded:
            return path, node, env, -1

        while True:
            if env.check_result() != GameResult.ONGOING:
                return path, node, env, -1

            # legal = env.get_valid_moves()
            # legal_int = [int(a) for a in legal]

            legal_int = self._focused_legal(env, radius=2) # for learning defence # original radius=5
            
            # # Progressive widening cap THIS SELECTS FROM THE TOP OF THE BOARD, CAN'T USE
            # M = 24 + int(2.0 * math.sqrt(sum(node.N.get(a,0) for a in legal_int)))
            # if len(legal_int) > M:
            #     # pick top-M by prior P
            #     legal_int = sorted(legal_int, key=lambda a: node.P.get(a, 0.0), reverse=True)[:M]

            total_N = sum(node.N.get(a, 0) for a in legal_int) + 1e-8
            best_action, best_score = None, -float("inf")
            for a in legal_int:
                # Q = node.Q.get(a, 0.0) if node.N.get(a, 0) > 0 else 0.0
                avg_q = np.mean([node.Q[a] for a in node.Q.keys()]) if node.Q else 0.0
                fpu = avg_q - 0.1  # tune lambda
                Q = node.Q.get(a, fpu) if node.N.get(a, 0) == 0 else node.Q[a]
                P = node.P.get(a, 0.0)
                U = self.c_puct * P * math.sqrt(total_N) / (1 + node.N.get(a, 0))
                s = Q + U
                if s > best_score:
                    best_score, best_action = s, a

            env.make_move(int(best_action))
            path.append((node, int(best_action)))

            if int(best_action) in node.children:
                node = node.children[int(best_action)]
                if not node.is_expanded:
                    return path, node, env, int(best_action)
            else:
                # create child placeholder; expansion will fill it
                child = MCTSNode(parent=node, action_from_parent=int(best_action))
                node.children[int(best_action)] = child
                return path, child, env, int(best_action)

    def _backup_path(self, path, leaf_value, leaf_env_current_player, path_first_node_player):
        # Backup: along full path
        if leaf_value is None:
            return  # nothing to back up
        # leaf_value is NN value at leaf; adjust sign if needed
        v = leaf_value if leaf_env_current_player == path_first_node_player else -leaf_value

        flip = 1
        for node, a in reversed(path):
            node.N[a] = node.N.get(a, 0) + 1
            node.W[a] = node.W.get(a, 0.0) + (v if flip == 1 else -v)
            node.Q[a] = node.W[a] / node.N[a]
            flip *= -1

    # ---- Expansion + Evaluation ----
    def _expand_and_evaluate(self, node: MCTSNode, env: Gomoku, parent_action: int):
        """
        Expand leaf node: query policy/value nets, update priors.
        Back up value to parent with perspective flip if needed.
        """
        # legal = env.get_valid_moves()
        # legal_int = [int(a) for a in legal]
        legal_int = self._focused_legal(env, radius=2) # for learning defence # original radius=5

        # # Progressive widening cap
        # M = 24 + int(2.0 * math.sqrt(sum(node.N.get(a,0) for a in legal_int)))
        # if len(legal_int) > M:
        #     # pick top-M by prior P
        #     legal_int = sorted(legal_int, key=lambda a: node.P.get(a, 0.0), reverse=True)[:M]

        state_t = self.encode_state(env.board, env.current_player)

        # with torch.no_grad():
        #     logits = self.policy_net(state_t)                 # [1, A]
        #     priors_all = F.softmax(logits, dim=-1).squeeze(0) # [A]
        #     value = float(self.value_net(state_t).item())     # scalar in [-1,1]

        with torch.no_grad():
            logits = self.policy_net(state_t)
            priors_all = F.softmax(logits, dim=-1).squeeze(0)
            value = float(self.value_net(state_t).item())

        # cache NN value on this node
        node.value_eval = value

        priors_np = priors_all.detach().cpu().numpy()
        masked_priors = np.zeros_like(priors_np, dtype=np.float32)
        if len(legal_int) > 0:
            masked_priors[legal_int] = priors_np[legal_int]
            s = masked_priors.sum()
            if s > 0:
                masked_priors /= s
            else:
                masked_priors[legal_int] = 1.0 / len(legal_int)

        # Initialize node if first time
        if not node.is_expanded:
            node.current_player = env.current_player
            for a in legal_int:
                node.P[a] = float(masked_priors[a])
                node.N[a] = 0
                node.W[a] = 0.0
                node.Q[a] = 0.0                
            node.is_expanded = True

        # Backup value along the edge (flip perspective if needed)
        if parent_action != -1:
            v = value if env.current_player == node.current_player else -value
            a = int(parent_action)
            node.N[a] = node.N.get(a, 0) + 1
            node.W[a] = node.W.get(a, 0.0) + v
            node.Q[a] = node.W[a] / node.N[a]

        # Create children lazily for all currently legal moves
        for a in legal_int:
            if a not in node.children:
                child = MCTSNode(parent=node, action_from_parent=a)
                child.current_player = -env.current_player
                node.children[a] = child

        # Ensure the child for parent_action exists even if it's not in legal anymore
        if parent_action != -1:
            a = int(parent_action)
            if a not in node.children:
                child = MCTSNode(parent=node, action_from_parent=a)
                child.current_player = -env.current_player
                node.children[a] = child
            return node.children[a]

        return None

    # ---- Simulation loop ----
    def run(self, root_env: Gomoku, add_root_noise: bool = True) -> Tuple[MCTSNode, np.ndarray]:
        """
        Run MCTS simulations starting from root_env state.
        Returns:
        - root node
        - policy target pi: visit-count distribution over actions (shape [BOARD_SIZE*BOARD_SIZE])
        """         
        cutoff_hits = 0
        cutoff_values = []

        root = MCTSNode(parent=None)

        legal_int = self._focused_legal(root_env, radius=2) # for learning defence # original radius=5

        state_t = self.encode_state(root_env.board, root_env.current_player)
        with torch.no_grad():
            root_logits = self.policy_net(state_t)
            priors_all = F.softmax(root_logits, dim=-1).squeeze(0)

        priors_np = priors_all.detach().cpu().numpy()
        masked_priors = np.zeros_like(priors_np, dtype=np.float32)
        if len(legal_int) > 0:
            masked_priors[legal_int] = priors_np[legal_int]
            s = masked_priors.sum()
            if s > 0:
                masked_priors /= s
            else:
                masked_priors[legal_int] = 1.0 / len(legal_int)

        root.current_player = root_env.current_player
        for a in legal_int:
            root.P[a] = float(masked_priors[a])
            root.N[a] = 0
            root.W[a] = 0.0
            root.Q[a] = 0.0
        root.is_expanded = True

        if add_root_noise and len(legal_int) > 0:
            root.add_dirichlet_noise(alpha=self.dirichlet_alpha, epsilon=self.dirichlet_eps)

        # --- Run simulations ---
        for _ in range(self.sims_per_move):
            sim_env = self._clone_env(root_env)

            # SELECT
            path, leaf, leaf_env, parent_action = self._select(root, sim_env)
            depth = len(path)

            # TERMINAL?
            if leaf_env.check_result() != GameResult.ONGOING:
                if leaf_env.find_any_five() == (None, []):
                    print("[MCTS] Warning: Terminal state reached but no winner found.")

                v_eval = self._terminal_value(leaf_env, leaf.current_player)
                self._backup_path(path, v_eval, leaf_env.current_player,
                                path[0][0].current_player if path else root.current_player)
                continue

            # EARLY CUTOFF
            if depth >= self.limit_rollout_depth:
                state_t = self.encode_state(leaf_env.board, leaf_env.current_player)
                with torch.no_grad():
                    v_eval = float(self.value_net(state_t).item())
                self._backup_path(path, v_eval, leaf_env.current_player,
                                path[0][0].current_player if path else root.current_player)
                cutoff_hits += 1
                cutoff_values.append(v_eval)
                continue

            # EXPAND + EVAL
            child = self._expand_and_evaluate(leaf, leaf_env, parent_action)
            v_eval = child.value_eval if child else leaf.value_eval
            self._backup_path(path, v_eval, leaf_env.current_player,
                            path[0][0].current_player if path else root.current_player)

        # --- Build policy target ---
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        total_visits = sum(root.N.get(a, 0) for a in legal_int)
        if total_visits == 0:
            for a in legal_int:
                pi[a] = root.P.get(a, 0.0)
        else:
            temp = max(self.temperature, 1e-6)
            for a in legal_int:
                pi[a] = (root.N.get(a, 0) ** (1.0 / temp))
        s = pi[legal_int].sum()
        if s > 0:
            pi[legal_int] /= s
        else:
            if len(legal_int) > 0:
                pi[legal_int] = 1.0 / len(legal_int)

        # Logging
        if cutoff_hits > 0:
            avg_cutoff_val = sum(cutoff_values) / cutoff_hits
            print(f"[MCTS] Cutoff hits this move: {cutoff_hits}/{self.sims_per_move}, "
                f"avg value_net prediction={avg_cutoff_val:.4f}")
        else:
            print(f"[MCTS] Cutoff hits this move: 0/{self.sims_per_move}")        
        if len(legal_int) > 0:
            visited_actions = [a for a in legal_int if root.N.get(a, 0) > 0]
            avg_q = float(np.mean([root.Q[a] for a in visited_actions])) if visited_actions else 0.0
            print(f"[MCTS] Average Q at root: {avg_q:.4f}")
            self.average_q_at_root = avg_q

            top = sorted(((a, root.N.get(a, 0)) for a in legal_int), key=lambda kv: kv[1], reverse=True)[:5]
            print("[PlayGame] Top 5 root actions:")
            for a, n in top:
                r, c = divmod(a, BOARD_SIZE)
                q_val = root.Q.get(a, 0.0)
                print(f"  ({r},{c}) -> action={a}, N={n}, Q={q_val:.4f}")

        return root, pi

    # ---- Utilities ----
    def _clone_env(self, env: Gomoku) -> Gomoku:
        """Create a deep copy of the environment state."""
        new_env = self.env_cls(size=env.size, win_len=env.win_len)
        new_env.board = env.board.copy()
        new_env.current_player = env.current_player
        new_env.moves_played = env.moves_played
        new_env.last_move = env.last_move
        return new_env

    def _terminal_value(self, env: Gomoku, reference_player: int) -> float:
        """
        Return terminal value from the perspective of reference_player.
        +1 if reference_player eventually wins, -1 if loses, 0 draw.
        """
        res = env.check_result()
        if res == GameResult.DRAW:
            return 0.0
        winner = BLACK if res == GameResult.BLACK_WIN else WHITE
        return 1.0 if winner == reference_player else -1.0
    

#   UNUSED FUNCTIONS
# -----------------------------
# Self-play with MCTS (collect training data)
# -----------------------------
def self_play_game_with_mcts(env: Gomoku, mcts: MCTS, temperature_by_move: int = 20):
    env.reset()
    samples = []
    move_idx = 0

    while True:
        result = env.check_result()
        if result != GameResult.ONGOING:
            break

        # Temperature schedule
        mcts.temperature = 1.0 if move_idx < temperature_by_move else 1e-3
        _, pi = mcts.run(env, add_root_noise=(move_idx == 0))

        legal = env.get_valid_moves()
        pi_masked = pi.copy()
        pi_masked[list(set(range(BOARD_SIZE*BOARD_SIZE)) - set(legal))] = 0.0
        s = pi_masked.sum()
        if s > 0:
            pi_masked /= s
        else:
            pi_masked[legal] = 1.0 / len(legal)

        # Record sample before move
        state_t = mcts.encode_state(env.board, env.current_player)
        samples.append((state_t.detach().cpu(), pi_masked.copy(), env.current_player))

        # Play move
        action = np.random.choice(np.arange(BOARD_SIZE*BOARD_SIZE), p=pi_masked)
        env.make_move(action)
        move_idx += 1

    # Assign z
    training_data = []
    if result == GameResult.DRAW:
        z_map = {BLACK: 0.0, WHITE: 0.0}
    else:
        winner = BLACK if result == GameResult.BLACK_WIN else WHITE
        z_map = {winner: 1.0, -winner: -1.0}
    for state_t, pi, player in samples:
        training_data.append((state_t, pi, z_map[player]))

    print(f"[SelfPlay] Game finished in {env.moves_played} moves, result={result.name}")
    return training_data

# -----------------------------
# Training step (AlphaZero-style)
# -----------------------------
def train_from_self_play_batch(policy_net, value_net,
                               optimizer_policy, optimizer_value,
                               batch,
                               lam_policy=1.0, lam_value=1.0, lam_reg=1e-4):
    states = torch.cat([s.to(DEVICE) for (s, _, _) in batch], dim=0)
    pis = torch.from_numpy(np.array([pi for (_, pi, _) in batch], dtype=np.float32)).to(DEVICE)
    zs = torch.tensor([z for (_, _, z) in batch], dtype=torch.float32, device=DEVICE).view(-1, 1)

    logits = policy_net(states)
    values = value_net(states)

    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(pis * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, zs)

    reg_loss = lam_reg * sum((p.norm(2) ** 2) for p in list(policy_net.parameters()) + list(value_net.parameters()))
    total_loss = lam_policy * policy_loss + lam_value * value_loss + reg_loss

    optimizer_policy.zero_grad()
    optimizer_value.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()) + list(value_net.parameters()), max_norm=5.0)
    optimizer_policy.step()
    optimizer_value.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total_loss.item()),
    }

# -----------------------------
# Self-play driver
# -----------------------------
def run_self_play_and_train(agent, num_self_play_games: int = 5, sims_per_move: int = 200):
    mcts = MCTS(
        env_cls=Gomoku,
        encode_state_fn=agent.encode_state,
        policy_net=agent.policy,
        value_net=agent.value,
        sims_per_move=sims_per_move,
        c_puct=2.0, # ORIGINAL VALUE 1.5
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    )

    for g in range(num_self_play_games):
        env = Gomoku()
        data = self_play_game_with_mcts(env, mcts, temperature_by_move=20)

        # Train immediately after each game
        stats = train_from_self_play_batch(
            policy_net=agent.policy,
            value_net=agent.value,
            optimizer_policy=agent.policy_opt,
            optimizer_value=agent.value_opt,
            batch=data,
            lam_policy=1.0,
            lam_value=1.0,
            lam_reg=1e-4,
        )

        # Update game counter
        agent.game_counter = getattr(agent, "game_counter", 0) + 1

        # Log stats with timestamp
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[Train] Game {agent.game_counter} stats: {stats}")
        with open("run_self_play_and_train.txt", "a") as f:
            f.write(f"Game {agent.game_counter} [{timestamp}]: "
                    f"policy_loss={stats['policy_loss']:.4f}, "
                    f"value_loss={stats['value_loss']:.4f}, "
                    f"total_loss={stats['total_loss']:.4f}\n")
