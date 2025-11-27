import tkinter as tk
from tkinter import messagebox

from Utils import GameResult, BLACK, WHITE, coord_to_index
from Gomoku import Gomoku

import os
os.chdir(r"C:\Users\...\Documents\PythonProjects\Gomoku DRL")

# -----------------------------
# GUI
# -----------------------------
class GomokuGUI:
    """
    Gomoku GUI using tkinter.
    - Draws board and stones
    - Handles mouse clicks
    - Maps clicks to intersections
    - Prevents illegal moves
    - Shows popup at game end
    """
    def __init__(self, env: Gomoku, cell_size: int = 40, margin: int = 20):
        self.env = env
        self.size = env.size
        self.cell_size = cell_size
        self.margin = margin
        self.window = tk.Tk()
        self.window.title("Gomoku")
        canvas_size = margin * 2 + cell_size * (self.size - 1)
        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg="burlywood")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.last_action = None

        self.draw_board()
        self.draw_stones()

    def draw_board(self):
        """Draw grid lines."""
        for i in range(self.size):
            x = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, x, self.margin + self.cell_size * (self.size - 1), x)
            self.canvas.create_line(x, self.margin, x, self.margin + self.cell_size * (self.size - 1))

    def draw_stones(self):
        """Draw stones based on env.board and highlight the last move."""
        # Clear previous stones and highlight
        self.canvas.delete("stone")
        self.canvas.delete("highlight")

        for r in range(self.size):
            for c in range(self.size):
                if self.env.board[r, c] != 0:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    radius = self.cell_size // 2 - 2
                    color = "black" if self.env.board[r, c] == BLACK else "white"
                    self.canvas.create_oval(
                        x - radius, y - radius, x + radius, y + radius,
                        fill=color, tags="stone"
                    )

        # Highlight the last move if available
        if self.env.last_move is not None:
            r, c = self.env.last_move
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            radius = self.cell_size // 2 - 2
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="red", width=2, tags="highlight"
            )

    def on_click(self, event):
        """Handle mouse click → nearest intersection mapping with enhancements."""
        # Prevent clicks if game already ended
        if self.env.check_result() != GameResult.ONGOING:
            print("[GUI] Click ignored: game already finished.")
            return

        # Map pixel coordinates to board indices (safer than round)
        # c = int((event.x - self.margin + self.cell_size / 2) // self.cell_size)
        # r = int((event.y - self.margin + self.cell_size / 2) // self.cell_size)
        c = round((event.x - self.margin) / self.cell_size)
        r = round((event.y - self.margin) / self.cell_size)

        if 0 <= r < self.size and 0 <= c < self.size:
            action = coord_to_index(r, c, self.size)
            if action in self.env.get_valid_moves():
                self.last_action = action
                prev_player = self.env.current_player
                self.env.make_move(action)
                self.draw_stones()

                # Log move to console                
                print(f"[GUI] Move played: row={r}, col={c}, player={prev_player}")                

                # Check for end-game after move
                result = self.env.check_result()
                if result != GameResult.ONGOING:
                    self.show_end_popup(result)
                    # Disable further clicks
                    self.canvas.unbind("<Button-1>")

    def get_user_action(self):
        """Return last valid action chosen by user, then reset."""
        action = self.last_action
        self.last_action = None
        return action

    def show_end_popup(self, result: GameResult):
        """Show popup with game result."""
        if result == GameResult.BLACK_WIN:
            messagebox.showinfo("Game Over", "Black wins!")
        elif result == GameResult.WHITE_WIN:
            messagebox.showinfo("Game Over", "White wins!")
        elif result == GameResult.DRAW:
            messagebox.showinfo("Game Over", "It's a draw!")

    def loop(self):
        """Run the tkinter main loop."""
        self.window.mainloop()

def extract_root_q(root_node):
    """
    Compute the average Q value at the root node.
    Returns a float in [-1, 1] representing expected outcome for current player.
    """    
    if not root_node.children:
        return root_node.value_eval

    # total_visits = sum(ch.N for ch in root_node.children)
    total_visits = sum(ch.N for ch in root_node.children.values())
    if total_visits == 0:
        return root_node.value_eval
    avg_q = sum((ch.W / ch.N) if ch.N > 0 else 0.0 for ch in root_node.children.values()) / len(root_node.children)
    # avg_q = sum((ch.W / ch.N) if ch.N > 0 else 0.0 for ch in root_node.children) / len(root_node.children)
    return avg_q
