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

from typing import List, Optional, Tuple, Dict

# from Gomoku import Gomoku
from GUI import GomokuGUI, Gomoku
from DRL import DRL
from MCTS import MCTS
from Utils import GameResult, Transition, coord_to_index, BLACK, WHITE

import os
os.chdir(r"C:\Users\XXX\Documents\PythonProjects\Gomoku") # Change to project directory

# -----------------------------
# Main game loop
# -----------------------------

import random

def random_first_move(board_size=15):
    row = random.randint(0, board_size - 1)
    col = random.randint(0, board_size - 1)
    return row, col

# # Example usage
# r, c = random_first_move()
# print(f"First stone placed at ({r}, {c})")

def play_game(env: Gomoku, agent: DRL,
              mode: str = "pve",
              human_is_black: bool = True,
              use_mcts: bool = False,
              gui: Optional[GomokuGUI] = None,
              deterministic_vs_human: bool = True) -> Tuple[GameResult, Dict[str, float]]:
    """
    Play a single game of Gomoku.
    - mode: "pve" (Player vs NN) or "eve" (NN vs NN)
    - human_is_black: True if player controls BLACK
    - use_mcts: True to use MCTS for NN moves
    - gui: GomokuGUI instance for interactive play (optional)
    - deterministic_vs_human: if True, NN plays deterministically when facing a human
    Returns (result, stats).
    """

    env.reset()
    transitions: List[Transition] = []

    while True:
        legal = env.get_valid_moves()
        result = env.check_result()
        if result != GameResult.ONGOING:
            break

        # Decide whose turn
        is_human_turn = (mode == "pve") and (
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
                    # root, pi = mcts.run(env)
                    # ...
                    mcts = MCTS(Gomoku, agent.encode_state, agent.policy, agent.value,
                                # sims_per_move=800, c_puct=1.0, temperature=1.0, dirichlet_alpha=0.2, dirichlet_eps=0.25)
                                sims_per_move=300, c_puct=3.0, temperature=2.0, dirichlet_alpha=0.7, dirichlet_eps=0.4)
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
                move_ok = env.make_move(action)
                if not move_ok:
                    transitions.append(Transition(
                        state=prev_board,
                        action=action,
                        reward=-1.0,
                        next_state=env.board.copy(),
                        done=False,
                        player=prev_player,
                        q_value=0.0,
                        pi=None,
                        z=None
                    ))
                    continue

                transitions.append(Transition(
                    state=prev_board,
                    action=action,
                    reward=0.0,
                    next_state=env.board.copy(),
                    done=False,
                    player=prev_player,
                    q_value=0.0,
                    pi=None,
                    z=None
                ))

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

# def play_game(env: Gomoku, agent: DRL,
#               mode: str = "pve",
#               human_is_black: bool = True,
#               use_mcts: bool = False,
#               gui: Optional[GomokuGUI] = None,
#               deterministic_vs_human: bool = True) -> Tuple[GameResult, Dict[str, float]]:
#     """
#     Play a single game of Gomoku.
#     - mode: "pve" (Player vs NN) or "eve" (NN vs NN)
#     - human_is_black: True if player controls BLACK
#     - use_mcts: True to use MCTS for NN moves
#     - gui: GomokuGUI instance for interactive play (optional)
#     - deterministic_vs_human: if True, NN plays deterministically when facing a human
#     Always returns (result, stats) where stats is a dict of training losses.
#     """


#     env.reset()
#     transitions: List[Transition] = []
#     mcts_samples: List[Tuple] = []

#     while True:
#         legal = env.get_valid_moves()
#         result = env.check_result()
#         if result != GameResult.ONGOING:
#             break

#         # Decide whose turn
#         is_human_turn = (mode == "pve") and (
#             (env.current_player == BLACK and human_is_black) or
#             (env.current_player == WHITE and not human_is_black)
#         )

#         if is_human_turn:
#             if gui:
#                 gui.draw_stones()
#                 action = None
#                 while action is None:
#                     gui.window.update()
#                     action = gui.get_user_action()
#                 # ensure action is legal
#                 if action not in legal or not env.make_move(action):
#                     continue
#             else:
#                 env.render()
#                 print("Enter move as 'row col' (0-based):")
#                 try:
#                     r, c = map(int, input().strip().split())
#                     action = coord_to_index(r, c, env.size)
#                 except Exception:
#                     print("Invalid input. Try again.")
#                     continue
#                 if action not in legal or not env.make_move(action):
#                     print("Illegal move. Try again.")
#                     continue

#             # Optional: evaluate q_value from value net at the previous perspective
#             # Here we set a neutral placeholder; you can swap to a network eval if desired
#             transitions.append(Transition(
#                 state=env.board.copy(),
#                 action=action,
#                 reward=0.0,
#                 next_state=env.board.copy(),
#                 done=False,
#                 player=WHITE if env.current_player == BLACK else BLACK,
#                 q_value=0.0  # or use agent.value(...) if you prefer
#             ))

#         else:
#             prev_board = env.board.copy()
#             prev_player = env.current_player  # capture perspective before move

#             if use_mcts:
#                 mcts = MCTS(Gomoku, agent.encode_state, agent.policy, agent.value,
#                             sims_per_move=400, c_puct=1.0)  # tune as needed

#                 # Run MCTS: ensure it returns a normalized π over all indices and masks illegal moves internally
#                 root, pi = mcts.run(env)

#                 # Sample only among legal moves to avoid residual mass on illegal indices
#                 pi_legal = np.array([pi[a] for a in legal], dtype=np.float32)
#                 s = pi_legal.sum()
#                 if s <= 0:
#                     # fallback uniform on legal moves
#                     pi_legal = np.ones(len(legal), dtype=np.float32) / len(legal)
#                 else:
#                     pi_legal /= s
                
#                 if mode == "pve" and deterministic_vs_human:
#                     # Deterministic: pick argmax among legal moves
#                     action = legal[np.argmax(pi_legal)]
#                 else:
#                     # Stochastic: sample according to distribution
#                     action = np.random.choice(legal, p=pi_legal)

#                 env.make_move(action)

#                 # AlphaZero sample (state at previous player perspective)
#                 state_t = agent.encode_state(prev_board, prev_player)
#                 mcts_samples.append((state_t.detach().cpu(), pi.copy(), prev_player))

#                 # DRL transition sample, attach root Q estimate safely
#                 # Prefer to have mcts.run compute and expose avg_q; if not, compute from dict stats:
#                 # avg_q = np.mean([root.Q[a] for a in root.P.keys() if root.N[a] > 0]) if any(root.N.values()) else 0.0
#                 avg_q = getattr(mcts, "average_q_at_root", None)
#                 if avg_q is None:
#                     # Conservative fallback: 0.0; or compute from root dicts if available
#                     try:
#                         counts = [root.N[a] for a in root.P.keys()]
#                         if sum(counts) > 0:
#                             avg_q = float(np.mean([root.Q[a] for a in root.P.keys() if root.N[a] > 0]))
#                         else:
#                             avg_q = 0.0
#                     except Exception:
#                         avg_q = 0.0

#                 transitions.append(Transition(
#                     state=prev_board,
#                     action=action,
#                     reward=0.0,
#                     next_state=env.board.copy(),
#                     done=False,
#                     player=prev_player,      # player who acted
#                     q_value=avg_q,
#                     pi=pi.copy(),            # full-length distribution (masked/normalized in train loop)
#                     z=None                   # to be filled post-game
#                 ))

#             else:
#                 # Pure NN policy: select among legal moves
#                 action = agent.select_action(env, legal)
#                 move_ok = env.make_move(action)
#                 if not move_ok:
#                     # Illegal move penalty
#                     q_est = 0.0
#                     try:
#                         # Optional: value estimate for additional signal
#                         v_t = agent.encode_state(prev_board, prev_player)
#                         with torch.no_grad():
#                             q_est = float(agent.value(v_t).item())
#                     except Exception:
#                         pass

#                     transitions.append(Transition(
#                         state=prev_board,
#                         action=action,
#                         reward=-1.0,
#                         next_state=env.board.copy(),
#                         done=False,
#                         player=WHITE if env.current_player == BLACK else BLACK,
#                         q_value=q_est
#                     ))
#                     continue

#                 # Normal transition: attach value estimate as q_value
#                 q_est = 0.0
#                 try:
#                     v_t = agent.encode_state(prev_board, prev_player)
#                     with torch.no_grad():
#                         q_est = float(agent.value(v_t).item())
#                 except Exception:
#                     pass

#                 transitions.append(Transition(
#                     state=prev_board,
#                     action=action,
#                     reward=0.0,
#                     next_state=env.board.copy(),
#                     done=False,
#                     player=WHITE if env.current_player == BLACK else BLACK,
#                     q_value=q_est
#                 ))

#         if gui:
#             gui.draw_stones()

#     # Game ended
#     if gui:
#         gui.draw_stones()
#     else:
#         env.render()
#     print(f"Result: {result.name}")

#     # --- DRL replay buffer training ---
#     for t in transitions:
#         t.done = True
#     DRL.compute_rewards(result, transitions)
#     print(f"[Debug] transitions collected: {len(transitions)}")
#     for t in transitions:
#         agent.store_transition(t)

#     # You can tune evolutionary params; keep them stable unless profiling
#     agent.evolve_transitions(num_parents=2, num_offspring=8, q_threshold=0.1)
#     drl_stats = agent.train_after_game()  # ensure dict of losses

#     # --- AlphaZero-style training ---
#     if use_mcts and mcts_samples:
#         if result == GameResult.DRAW:
#             z_map = {BLACK: 0.0, WHITE: 0.0}
#         else:
#             winner = BLACK if result == GameResult.BLACK_WIN else WHITE
#             z_map = {winner: 1.0, -winner: -1.0}

#         batch = []
#         for state_t, pi, player in mcts_samples:
#             z = z_map[player]
#             batch.append((state_t, pi, z))

#         az_stats = train_from_self_play_batch(agent.policy, agent.value,
#                                               agent.policy_opt, agent.value_opt,
#                                               batch)
#         print(f"AlphaZero-style training stats: {az_stats}")
#         stats = az_stats
#     else:
#         stats = drl_stats if drl_stats is not None else {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

#     return result, stats
