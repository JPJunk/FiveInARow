import numpy as np
import torch
import heapq

from dataclasses import dataclass

import MCTS

from Gomoku import Gomoku
from EnvAdapter import EnvAdapter

import os
os.chdir(r"C:\Users\XXX\Documents\PythonProjects\Gomoku") # Change to project directory

# -----------------------------
# Backwards MCTS
# -----------------------------

# @dataclass
# class Node:
#     state_hash: int
#     parent: 'Node' = None
#     action_from_parent: int = None
#     children: list = None  # list of (action, child_node)
#     N: int = 0
#     W: float = 0.0
#     priors: np.ndarray = None
#     value_eval: float = 0.0
#     state_tensor: torch.Tensor = None  # [C,H,W]

@dataclass
class Node:
    state_hash: int
    parent: 'Node' = None
    action_from_parent: int = None
    children: list = None  # list[(action:int, child:Node)]
    N: int = 0            # visits to this state
    W: float = 0.0        # total value at this state
    priors: np.ndarray = None
    value_eval: float = 0.0
    # state_tensor is removed; use agent.encode_state(env.board, env.current_player) instead


class TTEntry:
    def __init__(self, priors, value):
        self.priors = priors  # np.ndarray
        self.value = float(value)
        self.N = 0
        self.W = 0.0

class TranspositionTable:
    def __init__(self):
        self._map = {}
    def get(self, h): return self._map.get(h, None)
    def put_if_absent(self, h, entry): 
        if h not in self._map: self._map[h] = entry
        return self._map[h]
    def update_stats(self, h, dN, dW):
        e = self._map.get(h); 
        if e is not None: e.N += dN; e.W += dW
    def q(self, h):
        e = self._map.get(h); 
        return 0.0 if (e is None or e.N == 0) else e.W / e.N

class PromisingLeafQueue:
    def __init__(self, max_items=1024):
        self.heap = []
        self.max_items = max_items
    def push(self, state_hash, value):
        heapq.heappush(self.heap, (-value, state_hash))
        if len(self.heap) > self.max_items: heapq.heappop(self.heap)
    def pop_top(self): return heapq.heappop(self.heap) if self.heap else None

class TreeIndex:
    def __init__(self): self.by_hash = {}
    def register(self, node): self.by_hash[node.state_hash] = node
    def get(self, h): return self.by_hash.get(h, None)

def evaluate_with_nn(env, policy_net, value_net):
    """
    Evaluate the current environment state with NN.
    Returns (priors, value):
      - priors: numpy array of length board_size*board_size
      - value: float scalar in [-1,1]
    """
    # Convert env state to tensor [1,C,H,W]
    state_tensor = env.state_tensor().unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        # Policy head: logits over moves
        policy_logits = policy_net(state_tensor)  # shape [1, N]
        priors = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Value head: scalar prediction
        value_pred = value_net(state_tensor)      # shape [1,1] or [1]
        value = float(value_pred.squeeze().item())

    return priors, value

def select_child(node, c_puct, tt: TranspositionTable):
    if not node.children:
        return None
    total_N = sum(ch.N for _, ch in node.children) + 1e-8
    def uct(pair):
        a, ch = pair
        Q_local = (ch.W / ch.N) if ch.N > 0 else 0.0
        P_local = node.priors[a] if node.priors is not None else 0.0
        e = tt.get(ch.state_hash)
        Q_tt = tt.q(ch.state_hash) if e else 0.0
        P_tt = e.priors[a] if e is not None else 0.0
        alpha = 0.5 if ch.N == 0 else 0.2
        Q = (1 - alpha) * Q_local + alpha * Q_tt
        P = (1 - alpha) * P_local + alpha * P_tt
        U = c_puct * P * np.sqrt(total_N) / (1 + ch.N)
        return Q + U
    return max(node.children, key=uct)

def evaluate_with_nn(env, encode_state_fn, policy_net, value_net):
    state_t = encode_state_fn(env.board, env.current_player)
    with torch.no_grad():
        logits = policy_net(state_t)  # [1, A]
        priors = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        value = float(value_net(state_t).squeeze().item())
    return priors, value

def expand(node, env_adapter: EnvAdapter, encode_state_fn, policy_net, value_net, tt: TranspositionTable):
    h = EnvAdapter.hash_env(env_adapter.env)
    entry = tt.get(h)
    if entry is None:
        priors, value = evaluate_with_nn(env_adapter.env, encode_state_fn, policy_net, value_net)
        entry = tt.put_if_absent(h, TTEntry(priors=priors, value=value))
    else:
        priors, value = entry.priors, entry.value

    node.priors = priors
    node.value_eval = value
    node.children = []

    legal = env_adapter.env.get_valid_moves()
    for a in legal:
        env_adapter.push(a)
        child_hash = EnvAdapter.hash_env(env_adapter.env)
        child = Node(state_hash=child_hash, parent=node, action_from_parent=a,
                     children=[], N=0, W=0.0, priors=None, value_eval=0.0)
        node.children.append((a, child))
        env_adapter.pop()

    return node

def backup(path_nodes, leaf_value, tt: TranspositionTable):
    v = leaf_value
    for node in reversed(path_nodes):
        node.N += 1
        node.W += v
        tt.update_stats(node.state_hash, dN=1, dW=v)
        v = -v  # flip perspective

def on_leaf(node, value, queue: PromisingLeafQueue, v_thresh=0.6):
    if abs(value) >= v_thresh:
        queue.push(node.state_hash, abs(value))

def backward_expand_from_leaf(leaf_hash, tree_index: TreeIndex, env, tt, 
                              policy_net, value_net, sims_budget=32, c_puct=1.5):
    leaf = tree_index.get(leaf_hash)
    if leaf is None: return

    # Walk up; at each ancestor, prioritize low-visit siblings
    anc = leaf.parent
    while anc is not None and sims_budget > 0:
        siblings = [(a, ch) for (a, ch) in anc.children if ch is not leaf]
        # Score siblings by (low N, promising TT Q, decent prior)
        def score(pair):
            a, ch = pair
            e = tt.get(ch.state_hash)
            q = tt.q(ch.state_hash) if e else 0.0
            p = anc.priors[a] if anc.priors is not None else 0.0
            return (ch.N, -(q + 0.25 * p))  # low N first, then higher q/p
        siblings.sort(key=score)

        for a, ch in siblings:
            if ch.children is None or len(ch.children) == 0:
                # bring env to ancestor, then push action a
                env.restore_from_node(anc)  # implement: reconstruct env by path
                env.push(a)
                expand(ch, env, policy_net, value_net, tt)
            # Run 1–2 focused simulations from this sibling
            run_simulation_from_node(ch, env, tt, policy_net, value_net, c_puct=c_puct)
            sims_budget -= 1
            if sims_budget <= 0: break

        anc = anc.parent

def path_to_root(node):
    actions = []
    cur = node
    while cur.parent is not None:
        actions.append(cur.action_from_parent)
        cur = cur.parent
    return list(reversed(actions)), cur  # actions from root to node, root node

def run_simulation_from_node(start_node, root_env, encode_state_fn, tt, policy_net, value_net, c_puct=1.5):
    actions_to_start, root_node = path_to_root(start_node)
    env_adapter = EnvAdapter(Gomoku, base_env=MCTS._clone_env_static(root_env))
    env_adapter.restore_from_path(root_env, actions_to_start)

    path = [start_node]
    node = start_node
    # Selection
    while node.children and len(node.children) > 0:
        sel = select_child(node, c_puct, tt)
        if sel is None:
            break
        a, node = sel
        env_adapter.push(a)
        path.append(node)

    # Expand/evaluate at leaf
    if node.children is None or len(node.children) == 0:
        expand(node, env_adapter, encode_state_fn, policy_net, value_net, tt)

    v_leaf = node.value_eval
    backup(path, v_leaf, tt)

def backward_expand_from_leaf(leaf_hash, tree_index: TreeIndex, root_env, encode_state_fn, tt,
                              policy_net, value_net, sims_budget=32, c_puct=1.5):
    leaf = tree_index.get(leaf_hash)
    if leaf is None:
        return

    anc = leaf.parent
    while anc is not None and sims_budget > 0:
        siblings = [(a, ch) for (a, ch) in anc.children if ch is not leaf]
        def score(pair):
            a, ch = pair
            e = tt.get(ch.state_hash)
            q = tt.q(ch.state_hash) if e else 0.0
            p = anc.priors[a] if anc.priors is not None else 0.0
            return (ch.N, -(q + 0.25 * p))
        siblings.sort(key=score)

        for a, ch in siblings:
            if not ch.children:
                # restore env to ancestor state via path
                actions_to_anc, _ = path_to_root(anc)
                env_adapter = EnvAdapter(Gomoku, base_env=MCTS._clone_env_static(root_env))
                env_adapter.restore_from_path(root_env, actions_to_anc)
                env_adapter.push(a)
                expand(ch, env_adapter, encode_state_fn, policy_net, value_net, tt)
            run_simulation_from_node(ch, root_env, encode_state_fn, tt, policy_net, value_net, c_puct=c_puct)
            sims_budget -= 1
            if sims_budget <= 0:
                break
        anc = anc.parent