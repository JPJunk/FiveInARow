"""
Gomoku DRL Skeleton (PyTorch)
- Environment: Gomoku (15x15, two players, first to get exactly five in a row wins)
- Agent: DRL with policy/value network stubs, experience replay
- GUI: Optional stub (tkinter), interactive PVE
- Main: Player vs NN, NN vs NN. Training (backprop) after each game.
"""

from datetime import datetime

# Optional modules; ensure they exist or comment out
import evaluation, level_test

from Gomoku import Gomoku
from GUI import GomokuGUI
from DRL import DRL
from MCTS import MCTS
from Utils import GameResult, stages
from AgentPersistence import AgentPersistence
from PlayGame import play_game

import numpy as np
import torch

import numpy as np
import torch

import numpy as np
import torch

def test_policy_net_outputs(policy_net, board_size=15, device="cpu"):
    """
    Feed an empty board into the policy net and print the full 15x15 logits grid.
    """
    policy_net.eval()
    # Empty board: two channels, all zeros
    state = np.zeros((2, board_size, board_size), dtype=np.float32)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy_net(state_t).squeeze(0).cpu().numpy()

    # Reshape to board_size x board_size
    logits_grid = logits.reshape(board_size, board_size)

    print("Policy logits on empty board:")
    for r in range(board_size):
        row_str = " ".join(f"{logits_grid[r, c]:6.3f}" for c in range(board_size))
        print(row_str)

# def test_value_net_outputs(value_net, board_size=15, num_samples=5, device="cpu"):
#     """
#     Sample random Gomoku boards and print value_net predictions.
#     """
#     value_net.eval()
#     for i in range(num_samples):
#         # Random board: -1 = white, +1 = black, 0 = empty
#         board = np.random.choice([0, 1, -1], size=(board_size, board_size), p=[0.8, 0.1, 0.1])
        
#         # Encode as two channels: black stones, white stones
#         black = (board == 1).astype(np.float32)
#         white = (board == -1).astype(np.float32)
#         state = np.stack([black, white], axis=0)  # shape [2, board_size, board_size]
        
#         state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#         with torch.no_grad():
#             val = value_net(state_t).item()
#         print(f"Sample {i+1}: value_net output = {val:.4f}")

def test_value_net_outputs(value_net, board_size=15, num_samples=5, device="cpu"):
    """
    Sample random Gomoku boards and print value_net predictions.
    Also prints the board layout for each sample.
    """
    value_net.eval()
    for i in range(num_samples):
        # Random board: -1 = white, +1 = black, 0 = empty
        board = np.random.choice([0, 1, -1], size=(board_size, board_size), p=[0.8, 0.1, 0.1])

        # Encode as two channels: black stones, white stones
        black = (board == 1).astype(np.float32)
        white = (board == -1).astype(np.float32)
        state = np.stack([black, white], axis=0)  # shape [2, board_size, board_size]

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            val = value_net(state_t).item()

        print(f"\nSample {i+1}: value_net output = {val:.4f}")
        # Print board layout
        for r in range(board_size):
            row_str = " ".join(
                "X" if board[r, c] == 1 else "O" if board[r, c] == -1 else "."
                for c in range(board_size)
            )
            print(row_str)

def main():
    """
    Main entry point.
    - Choose mode: Player vs NN (pve) or NN vs NN (eve)
    - Player can choose Black or White
    - Toggle MCTS for NN moves
    - Backprop after each game
    - Results tracked and summarized
    - Agent state loaded at startup and saved after each game
    - GUI used for Player vs NN mode
    """
    env = Gomoku()
    agent = DRL()

    # Evaluation harness and scheduler (ensure these modules are available)
    try:
        eval_harness = evaluation.EvalHarness(
            env_cls=Gomoku,                # pass class, not instance
            policy_net=agent.policy.eval(),
            value_net=agent.value.eval(),
            mcts_cls=MCTS,
            board_size=env.size,
        )
        scheduler = level_test.LevelTestScheduler(agent, eval_harness, stages, eval_every_games=1)
        scheduler.current_idx = 0
    except Exception as e:
        print(f"[Eval] Skipping evaluation setup: {e}")
        eval_harness = None
        scheduler = None

    # Load previous agent state (method already handles exceptions)
    AgentPersistence.load(agent, "gomoku_agent.pth")

    test_value_net_outputs(agent.value, board_size=15, num_samples=10, device=agent.device)
    test_policy_net_outputs(agent.policy, board_size=15, device=agent.device)

    # Mode selection
    print("Choose mode:\n1) Player vs NN\n2) NN vs NN")
    choice = input("Enter 1 or 2: ").strip()
    mode = "pve" if choice == "1" else "eve"

    # MCTS toggle
    use_mcts = input("Use MCTS for NN moves? (y/n): ").strip().lower() == "y"

    # Player color selection
    human_is_black = True
    if mode == "pve":
        side = input("Play as Black(X) or White(O)? Enter B/W: ").strip().upper()
        human_is_black = (side == "B")

    # Number of games
    try:
        num_games = int(input("How many games to play? (default 5): ").strip() or "5")
    except Exception:
        num_games = 5

    # Results tracking
    results = {GameResult.BLACK_WIN: 0, GameResult.WHITE_WIN: 0, GameResult.DRAW: 0}

    # Create GUI if needed
    gui = GomokuGUI(env) if mode == "pve" else None

    # Game loop
    for g in range(1, num_games + 1):
        print(f"\n=== Game {g} ===")
        res, stats = play_game(
            env, agent,
            mode=mode,
            human_is_black=human_is_black,
            use_mcts=use_mcts,
            gui=gui,
        )
        results[res] += 1

        # Log
        timestamp = datetime.now().isoformat(timespec="seconds")
        agent.game_counter = getattr(agent, "game_counter", 0) + 1

        log_line = (
            f"Game {agent.game_counter} [{timestamp}]: result={res.name}, "
            f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
            f"value_loss={stats.get('value_loss', 0.0):.4f}, "
            f"total_loss={stats.get('total_loss', 0.0):.4f}"
        )
        with open("training_log.txt", "a") as f:
            f.write(log_line + "\n")
        print(f"[Log] {log_line}")

        # Auto-save after each game
        AgentPersistence.save(agent, "gomoku_agent.pth")

        # Optional scheduler promote step (if set up)
        if scheduler:
            try:
                promoted = scheduler.maybe_eval_and_promote()
                if promoted:
                    print(f"[Scheduler] Agent promoted to next training stage: {scheduler.current_stage.name}")
                    with open("training_log.txt", "a") as f:
                        f.write(f"Game {agent.game_counter} [{timestamp}]: PROMOTED to stage {scheduler.current_stage.name}\n")
                    break
            except Exception as e:
                print(f"[Scheduler] Skipped: {e}")

    # Summary
    print("\nSummary:")
    for k, v in results.items():
        print(f"{k.name}: {v}")

    if use_mcts:
        print("\nAlphaZero-style training was applied using MCTS visit counts.")
    else:
        print("\nClassic DRL replay buffer training was applied.")

if __name__ == "__main__":
    main()

# """
# Gomoku DRL Skeleton (PyTorch)
# - Environment: Gomoku (15x15, two players, first to get exactly five in a row wins)
# - Agent: DRL with policy/value network stubs, experience replay
# - GUI: Optional stub (tkinter/pygame), non-functional scaffolding
# - Main: Player vs NN, NN vs NN. Training (backprop) after each game.

# Notes:
# - This is a scaffold intended for incremental implementation.
# - Fill TODO sections with proper logic and training code.
# """

# from datetime import datetime

# import evaluation, level_test

# from Gomoku import Gomoku
# from GUI import GomokuGUI
# from DRL import DRL
# from MCTS import MCTS
# from Utils import GameResult, stages
# from AgentPersistence import AgentPersistence
# from PlayGame import play_game

# import os
# os.chdir(r"C:\Users\jpjun\Documents\PythonProjects\Gomoku DRL")

# # -----------------------------
# # Main game loop
# # -----------------------------

# def main():
#     """
#     Main entry point.
#     - Choose mode: Player vs NN (pve) or NN vs NN (eve)
#     - Player can choose to be Black or White
#     - User can toggle MCTS for NN moves
#     - Backpropagation is done after each game
#     - Results are tracked and summarized
#     - Agent state is loaded at startup and saved after each game
#     - GUI is used for Player vs NN mode
#     """
#     env = Gomoku()
#     agent = DRL()

#     # -----------------------------
#     # imports
#     # -----------------------------
#     eval_harness = evaluation.EvalHarness(env_cls=env, policy_net=agent.policy, value_net=agent.value, mcts_cls=MCTS, board_size=15)
#     scheduler = level_test.LevelTestScheduler(agent, eval_harness, stages, eval_every_games=1)
#     scheduler.current_idx = 0  # start at first stage


#     # Try to load previous agent state
#     try:
#         AgentPersistence.load(agent, "gomoku_agent.pth")
#         print("[Persistence] Loaded existing agent state.")
#     except Exception:
#         print("[Persistence] No saved agent found, starting fresh.")

#     # Mode selection
#     print("Choose mode:")
#     print("1) Player vs NN")
#     print("2) NN vs NN")
#     choice = input("Enter 1 or 2: ").strip()
#     mode = "pve" if choice == "1" else "eve"

#     # MCTS toggle
#     use_mcts = input("Use MCTS for NN moves? (y/n): ").strip().lower() == "y"

#     # Player color selection
#     human_is_black = True
#     if mode == "pve":
#         side = input("Play as Black(X) or White(O)? Enter B/W: ").strip().upper()
#         human_is_black = (side == "B")

#     # Number of games
#     num_games = 5
#     try:
#         num_games = int(input("How many games to play? (default 5): ").strip())
#     except Exception:
#         pass

#     # Results tracking
#     results = {GameResult.BLACK_WIN: 0, GameResult.WHITE_WIN: 0, GameResult.DRAW: 0}

#     # Create GUI if needed
#     gui = GomokuGUI(env) if mode == "pve" else None

#     # Game loop
#     for g in range(1, num_games + 1):
#         print(f"\n=== Game {g} ===")
#         res, stats = play_game(env, agent,
#                         mode=mode,
#                         human_is_black=human_is_black,
#                         use_mcts=use_mcts,
#                         gui=gui)
#         results[res] += 1

#         # Append to log file
#         timestamp = datetime.now().isoformat(timespec="seconds")
#         agent.game_counter = getattr(agent, "game_counter", 0) + 1

#         log_line = f"Game {agent.game_counter} [{timestamp}]: result={res.name}"
#         log_line += (f", policy_loss={stats.get('policy_loss', 0):.4f}, "
#                     f"value_loss={stats.get('value_loss', 0):.4f}, "
#                     f"total_loss={stats.get('total_loss', 0):.4f}")

#         with open("training_log.txt", "a") as f:
#             f.write(log_line + "\n")

#         print(f"[Log] {log_line}")

#         # Auto-save after each game
#         try:
#             AgentPersistence.save(agent, "gomoku_agent.pth")
#             print(f"[Persistence] Agent state saved after game {g}.")
#         except Exception as e:
#             print(f"[Persistence] Failed to save agent after game {g}: {e}")

#         # after each game and training step:
#         # promoted = scheduler.maybe_eval_and_promote()
#         # if promoted:
#         #     # Step completed, agent promoted to next stage
#         #     print(f"[Scheduler] Agent promoted to next training stage: {scheduler.current_stage.name}")
#         #     log_line = f"Game {agent.game_counter} [{timestamp}]: PROMOTED to stage {scheduler.current_stage.name}"
#         #     with open("training_log.txt", "a") as f:
#         #         f.write(log_line + "\n")

#         #     print(f"[Log] {log_line}")
#         #     # Stop further training for now
#         #     break

#     # Summary
#     print("\nSummary:")
#     for k, v in results.items():
#         print(f"{k.name}: {v}")

#     if use_mcts:
#         print("\nAlphaZero-style training was applied using MCTS visit counts.")
#     else:
#         print("\nClassic DRL replay buffer training was applied.")

# if __name__ == "__main__":
#     main()