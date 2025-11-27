import MCTS

# NOT READY YET

class EnvAdapter:
    def __init__(self, env_cls, base_env=None):
        self.env = base_env or env_cls()
        self.history = []  # stack of (action, last_move, current_player)

    def push(self, action):
        prev_last = self.env.last_move
        prev_player = self.env.current_player
        ok = self.env.make_move(action)
        if ok:
            self.history.append((action, prev_last, prev_player))
        return ok

    def pop(self):
        if not self.history:
            return False
        action, prev_last, prev_player = self.history.pop()
        # undo the move we just made
        r, c = divmod(action, self.env.size)
        self.env.board[r, c] = 0
        self.env.moves_played -= 1
        self.env.current_player = prev_player
        self.env.last_move = prev_last
        return True

    def restore_from_path(self, root_env, actions):
        # reset to root_env and re-apply actions in order
        self.env = MCTS._clone_env_static(root_env)
        self.history = []
        for a in actions:
            self.push(a)

    @staticmethod
    def hash_env(env):
        return env.hash()  # uses Gomoku.hash()

    @staticmethod
    def encode_state(encode_fn, env):
        return encode_fn(env.board, env.current_player)