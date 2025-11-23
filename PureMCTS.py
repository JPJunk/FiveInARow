import math
import random
import numpy as np

from Utils import GameResult, BLACK, WHITE
from Gomoku import Gomoku

class PureMCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        # Edge stats keyed by action
        self.P = {}                 # prior over actions (uniform here)
        self.N = {}                 # visit counts per action
        self.W = {}                 # total value per action
        self.Q = {}                 # mean value per action
        self.children = {}          # action -> child node
        self.is_expanded = False
        self.current_player = None  # set on expansion

class PureMCTS:
    """
    Pure MCTS without neural networks, for baseline comparisons.
    Uses uniform priors over legal moves and random rollouts for leaf evaluation.
    """
    def __init__(self, env_cls=Gomoku, sims_per_move: int = 500, c_puct: float = 1.0, temperature: float = 1.0):
        self.env_cls = env_cls
        self.sims_per_move = sims_per_move
        self.c_puct = c_puct
        self.temperature = temperature

    def _random_rollout_value(self, env: Gomoku, max_steps=128) -> float:
        """
        Play random legal moves until terminal or max_steps.
        Return value from perspective of player at leaf when called.
        """
        steps = 0
        # We will mutate env during simulation; ensure caller restores after.
        while env.check_result() == GameResult.ONGOING and steps < max_steps:
            legal = env.get_valid_moves()
            if not legal:
                break
            env.make_move(random.choice(legal))
            steps += 1
        res = env.check_result()
        if res == GameResult.DRAW:
            return 0.0
        # Winner relative to player-to-move at the start of rollout? We backed up along edges with perspective flips,
        # so returning +1 for winner (BLACK/WHITE) and -1 otherwise is sufficient here.
        winner = BLACK if res == GameResult.BLACK_WIN else WHITE
        # We want value for the player who just moved into this leaf (node.current_player at expansion).
        # The backup will flip along the path, so simply use +1/-1 w.r.t winner.
        return 1.0 if winner == BLACK else -1.0  # sign will be corrected by backup flips

    def _expand_with_uniform_priors(self, node: PureMCTSNode, env: Gomoku):
        legal = env.get_valid_moves()
        node.current_player = env.current_player
        if not legal:
            node.is_expanded = True
            return
        # Uniform priors over legal moves
        p = 1.0 / len(legal)
        for a in legal:
            node.P[a] = p
            node.N[a] = 0
            node.W[a] = 0.0
            node.Q[a] = 0.0
            # Create child lazily when selected
        node.is_expanded = True

    def _select(self, node: PureMCTSNode, env: Gomoku):
        """
        PUCT selection until reaching a leaf (unexpanded node) or a new child to create.
        Returns (leaf_node, parent_node, action_from_parent or -1)
        """
        # If not expanded yet, stop here
        if not node.is_expanded:
            return node, None, -1

        while True:
            result = env.check_result()
            if result != GameResult.ONGOING:
                # terminal: no action from parent
                return node, None, -1

            legal = env.get_valid_moves()
            if not legal:
                return node, None, -1

            total_N = sum(node.N.get(a, 0) for a in legal) + 1e-8
            best_score = -float("inf")
            best_action = None

            for a in legal:
                Q = node.Q.get(a, 0.0) if node.N.get(a, 0) > 0 else 0.0
                P = node.P.get(a, 0.0)
                U = self.c_puct * P * math.sqrt(total_N) / (1 + node.N.get(a, 0))
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_action = a

            # Apply best action
            env.make_move(best_action)

            # Descend if child exists, otherwise we will expand/evaluate at this new child
            if best_action in node.children:
                node = node.children[best_action]
                # If child is not expanded, stop and expand it
                if not node.is_expanded:
                    return node, node.parent, node.action_from_parent
                # continue loop
            else:
                # Create child placeholder; expand/evaluate next
                child = PureMCTSNode(parent=node, action_from_parent=best_action)
                node.children[best_action] = child
                return child, node, best_action

    def _evaluate_leaf(self, env: Gomoku) -> float:
        # Evaluate via random rollout; after evaluation we must undo moves to return to parent state.
        return self._random_rollout_value(env, max_steps=128)

    def _backup(self, parent: PureMCTSNode, action: int, v: float):
        # Update edge stats on parent for action
        parent.N[action] = parent.N.get(action, 0) + 1
        parent.W[action] = parent.W.get(action, 0.0) + v
        parent.Q[action] = parent.W[action] / parent.N[action]

    def run(self, root_env: Gomoku):
        root = PureMCTSNode()
        root.current_player = root_env.current_player
        self._expand_with_uniform_priors(root, root_env)

        for _ in range(self.sims_per_move):
            # Clone environment by copying fields
            sim = self.env_cls(size=root_env.size, win_len=root_env.win_len)
            sim.board = root_env.board.copy()
            sim.current_player = root_env.current_player
            sim.moves_played = root_env.moves_played
            sim.last_move = root_env.last_move

            # Selection (creates a new child or stops at unexpanded node)
            leaf, parent, parent_action = self._select(root, sim)

            # Terminal?
            if sim.check_result() != GameResult.ONGOING:
                # Terminal value from leaf perspective; flip sign once when backing up
                res = sim.check_result()
                v = 0.0 if res == GameResult.DRAW else (1.0 if (res == GameResult.BLACK_WIN) else -1.0)
                if parent is not None and parent_action != -1:
                    self._backup(parent, parent_action, v)
                # Undo moves made during selection
                continue

            # Expand leaf with uniform priors
            self._expand_with_uniform_priors(leaf, sim)

            # Evaluate leaf
            v_leaf = self._evaluate_leaf(sim)

            # Backup to parent edge (flip sign according to one ply)
            if parent is not None and parent_action != -1:
                # Value returned is from the perspective of the player who would receive the payoff at leaf.
                # Since parent_action moved from parent into leaf, we flip sign once for backup.
                self._backup(parent, parent_action, v_leaf)

            # Undo rollout moves: reconstruct sim back to root_env by re-copying
            # (selection mutated sim; we simply discard it and proceed)
            # Next simulation will reconstruct a fresh sim from root_env

        # Build policy π from root visit counts
        legal = root_env.get_valid_moves()
        pi = np.zeros(root_env.size * root_env.size, dtype=np.float32)
        total_visits = sum(root.N.get(a, 0) for a in legal)
        if total_visits == 0:
            for a in legal:
                pi[a] = root.P.get(a, 0.0)
        else:
            temp = max(self.temperature, 1e-6)
            for a in legal:
                pi[a] = (root.N.get(a, 0) ** (1.0 / temp))
            s = pi[legal].sum()
            if s > 0:
                pi[legal] /= s
            else:
                pi[legal] = 1.0 / len(legal)

        return root, pi

    def choose_move(self, env: Gomoku):
        root, pi = self.run(env)
        legal = env.get_valid_moves()
        # Deterministic: argmax visit count
        if legal:
            pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
            action = legal[int(np.argmax(pi_legal))]
            return action
        # Fallback
        return random.choice(legal) if legal else None