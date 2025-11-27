"""
Gomoku DRL Skeleton (PyTorch)
- Environment: Gomoku (15x15, two players, first to get exactly five in a row wins)
- Agent: DRL with policy/value network stubs, experience replay
- GUI: Optional stub (tkinter/pygame), non-functional scaffolding
- Main: Player vs NN, NN vs NN. Training (backprop) after each game.

Notes:
- This is a scaffold intended for incremental implementation.
- Fill TODO sections with proper logic and training code.
"""

import numpy as np
import torch

from typing import List, Optional, Tuple, Dict

# from Gomoku import Gomoku
from GUI import GomokuGUI, Gomoku
from DRL import DRL
from MCTS import MCTS
from Utils import GameResult, Transition, coord_to_index, BLACK, WHITE

import os
os.chdir(r"C:\Users\...\Documents\PythonProjects\Gomoku DRL")

# -----------------------------
# Main game loop
# -----------------------------

import random

def random_first_move(board_size=15):
    row = random.randint(0, board_size - 1)
    col = random.randint(0, board_size - 1)
    return row, col

def play_game(env: Gomoku, agent: DRL,
              mode: str = "pve",
              human_is_black: bool = True,
              use_mcts: bool = False,
              gui: Optional[GomokuGUI] = None,
              deterministic_vs_human: bool = True) -> Tuple[GameResult, Dict[str, float]]:
    """
    Play a single game of Gomoku.
    Modes:
      - "pve": Player vs NN (optionally with MCTS)
      - "eve": NN vs NN
      - "pvmcts": Player vs pure MCTS
      - "nn_vs_mcts": Deterministic NN vs MCTS
      - "pve_policy": Player vs raw policy_net
      - "pve_value": Player vs raw value_net
    """

    env.reset()
    transitions: List[Transition] = []

    while True:
        legal = env.get_valid_moves()
        result = env.check_result()
        if result != GameResult.ONGOING:
            break

        # --- Human turn check ---
        is_human_turn = (mode.startswith("pve") or mode == "pvmcts") and (
            (env.current_player == BLACK and human_is_black) or
            (env.current_player == WHITE and not human_is_black)
        )

        if is_human_turn:
            if gui:
                gui.draw_stones()
                action = None
                while action is None:
                    gui.window.update()
                    action = gui.get_user_action()
                if action not in legal or not env.make_move(action):
                    continue
            else:
                env.render()
                print("Enter move as 'row col' (0-based):")
                try:
                    r, c = map(int, input().strip().split())
                    action = coord_to_index(r, c, env.size)
                except Exception:
                    print("Invalid input. Try again.")
                    continue
                if action not in legal or not env.make_move(action):
                    print("Illegal move. Try again.")
                    continue

            transitions.append(Transition(
                state=env.board.copy(),
                action=action,
                reward=0.0,
                next_state=env.board.copy(),
                done=False,
                player=WHITE if env.current_player == BLACK else BLACK,
                q_value=0.0,
                pi=None,
                z=None
            ))

        else:
            prev_board = env.board.copy()
            prev_player = env.current_player

            # --- Branch by mode ---
            if mode == "pvmcts" or mode == "nn_vs_mcts":
                # Always use MCTS for this side
                mcts = MCTS(Gomoku, agent.encode_state, agent.policy, agent.value,
                            sims_per_move=500, c_puct=3.0, temperature=2.0,
                            dirichlet_alpha=0.7, dirichlet_eps=0.4)
                root, pi = mcts.run(env)
                pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
                pi_legal /= pi_legal.sum() if pi_legal.sum() > 0 else len(pi_legal)
                action = np.random.choice(legal, p=pi_legal)
                env.make_move(action)
                transitions.append(Transition(state=prev_board, action=action,
                                              reward=0.0, next_state=env.board.copy(),
                                              done=False, player=prev_player,
                                              q_value=getattr(mcts, "average_q_at_root", 0.0),
                                              pi=pi.copy(), z=None))

            elif mode == "pve_policy":
                # Direct argmax from policy_net
                state_t = agent.encode_state(env.board, env.current_player)
                with torch.no_grad():
                    logits = agent.policy(state_t).cpu().numpy().flatten()
                mask = np.full_like(logits, -np.inf)
                mask[legal] = logits[legal]
                action = int(np.argmax(mask))
                env.make_move(action)
                transitions.append(Transition(state=prev_board, action=action,
                                              reward=0.0, next_state=env.board.copy(),
                                              done=False, player=prev_player,
                                              q_value=0.0, pi=None, z=None))

            elif mode == "pve_value":
                # Pick move maximizing value_net prediction
                best_action, best_val = None, -float("inf")
                for a in legal:
                    tmp_env = env.clone()
                    tmp_env.make_move(a)
                    state_t = agent.encode_state(tmp_env.board, tmp_env.current_player)
                    with torch.no_grad():
                        val = float(agent.value(state_t).item())
                    if val > best_val:
                        best_val, best_action = val, a
                action = best_action
                env.make_move(action)
                transitions.append(Transition(state=prev_board, action=action,
                                              reward=0.0, next_state=env.board.copy(),
                                              done=False, player=prev_player,
                                              q_value=best_val, pi=None, z=None))

            else:
                # Default NN branch (pve/eve with optional MCTS toggle)
                if use_mcts:
                    if env.moves_played == 0:
                        board_size = env.size
                        # Random first stone
                        row = random.randint(0, board_size - 1)
                        col = random.randint(0, board_size - 1)
                        action = row * board_size + col
                        env.make_move(action)

                        # For replay buffer consistency, create a one-hot pi
                        pi = np.zeros(board_size * board_size, dtype=np.float32)
                        pi[action] = 1.0

                        transitions.append(Transition(
                            state=prev_board,
                            action=action,
                            reward=0.0,
                            next_state=env.board.copy(),
                            done=False,
                            player=prev_player,
                            q_value=0.0,   # no search yet, so Q is undefined; set 0
                            pi=pi,
                            z=None
                        ))
                    else:
                        # Normal MCTS branch (your existing code)
                        mcts = MCTS(Gomoku, agent.encode_state, agent.policy, agent.value,
                                    # sims_per_move=800, c_puct=1.0, temperature=1.0, dirichlet_alpha=0.2, dirichlet_eps=0.25)
                                    sims_per_move=500, c_puct=3.0, temperature=2.0, dirichlet_alpha=0.7, dirichlet_eps=0.4)
                        root, pi = mcts.run(env)

                        # Normalize over legal moves
                        pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
                        s = pi_legal.sum()
                        if s <= 0:
                            pi_legal = np.ones(len(legal), dtype=np.float32) / len(legal)
                        else:
                            pi_legal /= s

                        # --- Entropy and non‑zero count of π_legal ---
                        entropy = -np.sum(pi_legal * np.log(pi_legal + 1e-12))
                        nonzero_count = np.count_nonzero(pi_legal > 0)
                        print(f"[PlayGame] π_legal entropy={entropy:.4f}, non‑zero entries={nonzero_count}/{len(pi_legal)}")

                        if mode == "pve" and deterministic_vs_human:
                            action = legal[np.argmax(pi_legal)]
                        else:
                            action = np.random.choice(legal, p=pi_legal)

                        selected_action = action  # from your later choice
                        r_sel, c_sel = divmod(selected_action, board_size)
                        print(f"[PlayGame] Selected move: ({r_sel},{c_sel}) -> action={selected_action}")

                        env.make_move(action)

                        avg_q = getattr(mcts, "average_q_at_root", 0.0)

                        transitions.append(Transition(
                            state=prev_board,
                            action=action,
                            reward=0.0,
                            next_state=env.board.copy(),
                            done=False,
                            player=prev_player,
                            q_value=avg_q,
                            pi=pi.copy(),   # full distribution over all moves
                            z=None          # filled after game ends
                        ))

                else:
                    action = agent.select_action(env, legal)
                    env.make_move(action)
                    transitions.append(Transition(state=prev_board, action=action,
                                                  reward=0.0, next_state=env.board.copy(),
                                                  done=False, player=prev_player,
                                                  q_value=0.0, pi=None, z=None))

        if gui:
            gui.draw_stones()


    # Game ended
    if gui:
        gui.draw_stones()
    else:
        env.render()
    print(f"Result: {result.name}")

    # Mark transitions done and assign rewards
    for t in transitions:
        t.done = True
    DRL.compute_rewards(result, transitions)

    # Assign z for AlphaZero training
    if result == GameResult.DRAW:
        z_map = {BLACK: 0.0, WHITE: 0.0}
    else:
        winner = BLACK if result == GameResult.BLACK_WIN else WHITE
        z_map = {winner: 1.0, -winner: -1.0}
    for t in transitions:
        t.z = z_map[t.player]

    # Store transitions
    for t in transitions:
        agent.store_transition(t)

    agent.evolve_transitions(num_parents=2, num_offspring=8, q_threshold=0.1)

    # Choose training style
    if use_mcts:
        stats = agent.train_after_game_az()  # AlphaZero imitation
    else:
        stats = agent.train_after_game()     # old RL style

    return result, stats



    # --- End of game bookkeeping (same as before) ---
