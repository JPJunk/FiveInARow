import numpy as np
import hashlib

from datetime import datetime
from typing import List, Optional, Tuple

from Utils import GameResult, BOARD_SIZE, WIN_LENGTH, BLACK, WHITE, coord_to_index, index_to_coord

import os
os.chdir(r"C:\Users\...\Documents\PythonProjects\Gomoku DRL")

# -----------------------------
# Gomoku environment
# -----------------------------
class Gomoku:
    """
    Core Gomoku environment.
    Responsibilities:
    - Track board state
    - Validate moves (empty intersection, correct turn)
    - Check win condition (exactly five in a row, horizontal/vertical/diagonal)
    - Provide legal moves
    - Reset & render
    """
    def __init__(self, size: int = BOARD_SIZE, win_len: int = WIN_LENGTH):
        self.size = size
        self.win_len = win_len
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = BLACK
        self.moves_played: int = 0
        self.last_move: Optional[Tuple[int, int]] = None

    def reset(self):
        """Reset to initial state."""
        self.board.fill(0)
        self.current_player = BLACK  # Black always starts
        self.moves_played = 0
        self.last_move = None

    def hash(self) -> int:
        """Return a reproducible hash of the current board + player turn.
        2. hash() reproducibility
            You’re using self.board.tobytes() + bytes([player_byte]). 
            That’s fine, but note: np.int8 can produce different byte orders 
            across platforms. If you want cross‑platform reproducibility, 
            consider .astype(np.int8).flatten().tobytes() to enforce consistent layout.
        """
        # Flatten board and include current player
        player_byte = 0 if self.current_player == BLACK else 1
        data = self.board.tobytes() + bytes([player_byte])
        return int(hashlib.sha256(data).hexdigest(), 16)

    def get_valid_moves(self) -> List[int]:
        """Return list of legal move indices (empty intersections)."""
        # valid = []
        # # Simple version: any empty intersection is valid
        # for r in range(self.size):
        #     for c in range(self.size):
        #         if self.board[r, c] == 0:
        #             valid.append(coord_to_index(r, c, self.size))
        # return valid
        # OR
        rows, cols = np.where(self.board == 0)
        return [coord_to_index(r, c, self.size) for r, c in zip(rows, cols)]

    # def make_move(self, action: int) -> bool:
    #     """
    #     Place a stone for the current player at 'action' index.
    #     Returns True if move is valid and applied, else False.
    #     """
    #     r, c = index_to_coord(action, self.size)
    #     if r < 0 or r >= self.size or c < 0 or c >= self.size:
    #         return False
    #     if self.board[r, c] != 0:
    #         return False
    #     self.board[r, c] = self.current_player
    #     self.last_move = (r, c)
    #     self.moves_played += 1
    #     # Switch turn
    #     self.current_player = WHITE if self.current_player == BLACK else BLACK
    #     return True

    def make_move(self, action: int) -> bool:
        r, c = divmod(action, self.size)
        if self.board[r, c] != 0:
            return False
        # 1) place stone
        self.board[r, c] = self.current_player
        # 2) update last_move and moves_played
        self.last_move = (r, c)
        self.moves_played += 1
        # 3) check result NOW, before toggling
        self.result = self.check_result()
        # 4) toggle player only if game continues
        if self.result == GameResult.ONGOING:
            self.current_player = WHITE if self.current_player == BLACK else BLACK
        return True

    def undo_move(self) -> bool:
        """Undo the last move and restore the previous player."""
        if self.last_move is None:
            return False
        r, c = self.last_move
        self.board[r, c] = 0
        self.moves_played -= 1
        # Flip back to the player who made the undone move
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self.last_move = None
        return True

    # def check_result(self) -> GameResult:
    #     """
    #     Check if the game is won or drawn.
    #     ⚠️ Currently counts >= win_len as a win (free Gomoku).
    #     For strict Renju, enforce exactly win_len and disallow overlines.
    #     """
    #     if self.last_move is None:
    #         return GameResult.ONGOING
    #     r, c = self.last_move
    #     player = self.board[r, c]

    #     directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    #     def count_dir(dr, dc):
    #         cnt = 1
    #         rr, cc = r + dr, c + dc
    #         while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
    #             cnt += 1
    #             rr += dr; cc += dc
    #         rr, cc = r - dr, c - dc
    #         while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
    #             cnt += 1
    #             rr -= dr; cc -= dc
    #         return cnt

    #     for dr, dc in directions:
    #         length = count_dir(dr, dc)
    #         if length >= self.win_len:
    #             return GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN

    #     if self.moves_played >= self.size * self.size:
    #         return GameResult.DRAW
    #     return GameResult.ONGOING

    def check_result(self) -> GameResult:
        if self.last_move is None:
            return GameResult.ONGOING
        r, c = self.last_move
        player = self.board[r, c]

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def count_dir(dr, dc):
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                cnt += 1
                rr += dr; cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                cnt += 1
                rr -= dr; cc -= dc
            return cnt

        for dr, dc in directions:
            length = count_dir(dr, dc)
            if length >= self.win_len:
                # DEBUG: reveal winning line coordinates
                win_coords = []
                rr, cc = r, c
                while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                    win_coords.append((rr, cc))
                    rr += dr; cc += dc
                rr, cc = r - dr, c - dc
                left_coords = []
                while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                    left_coords.append((rr, cc))
                    rr -= dr; cc -= dc
                win_coords = list(reversed(left_coords)) + win_coords

                print(f"[ResultDebug] last_move={self.last_move}, player={'WHITE' if player==WHITE else 'BLACK'}, "
                    f"dir={(dr,dc)}, length={length}, line={win_coords}")

                return GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN

        if self.moves_played >= self.size * self.size:
            return GameResult.DRAW
        return GameResult.ONGOING

    def find_any_five(self, win_len=5):
        dirs = [(1,0),(0,1),(1,1),(1,-1)]
        for r in range(self.size):
            for c in range(self.size):
                player = self.board[r,c]
                if player == 0: continue
                for dr,dc in dirs:
                    coords = [(r,c)]
                    rr,cc = r+dr, c+dc
                    while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr,cc] == player:
                        coords.append((rr,cc))
                        if len(coords) >= win_len:
                            return player, coords
                        rr += dr; cc += dc
        return None, []

# UNUSED FUNCTIONS FOR INTERFACE CONSISTENCY
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        return self.check_result() != GameResult.ONGOING

    def current_player_str(self) -> str:
        """Return 'black' or 'white' for current player."""
        return "black" if self.current_player == BLACK else "white"

    def step(self, action: int) -> bool:
        """Alias for make_move to match expected interface."""
        return self.make_move(action)

    def render(self):
        """Simple text rendering (X for Black, O for White, . for empty)."""
        rows = []
        for r in range(self.size):
            row_chars = []
            for c in range(self.size):
                val = self.board[r, c]
                if val == BLACK:
                    row_chars.append("X")
                elif val == WHITE:
                    row_chars.append("O")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        print("\n".join(rows))
        print(f"Current player: {'BLACK(X)' if self.current_player == BLACK else 'WHITE(O)'}")

    def log_result(self, result: GameResult, filename="game_log.txt"):
        with open(filename, "a") as f:
            # f.write(f"Game finished after {self.moves_played} moves: {result.name}\n")
            f.write(f"[{datetime.now().isoformat()}] Game finished after {self.moves_played} moves: {result.name}\n")
