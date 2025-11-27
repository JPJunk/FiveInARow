import random
from collections import Counter

# NOT USED, NEEDS BETTER MATCH SETUP FROM MAIN.PY

class EvalHarness:
    def __init__(self, env_cls, policy_net, value_net, mcts_cls, board_size=15):
        self.env_cls = env_cls
        self.policy = policy_net.eval()  # ensure eval mode
        self.value = value_net.eval()
        self.mcts_cls = mcts_cls
        self.board_size = board_size

    def play_match(self, sims_agent: int, sims_opponent: int, agent_color: str = "black", seed: int = None):
        """Play a single game: agent (NN+MCTS) vs opponent (pure MCTS baseline)."""
        if seed is not None:
            random.seed(seed)

        env = self.env_cls(board_size=self.board_size)
        env.reset()

        # Agent: NN-assisted MCTS
        agent_mcts = self.mcts_cls(
            env_cls=self.env_cls,
            encode_state_fn=lambda board, player: self.policy.encode_state(board, player) if hasattr(self.policy, "encode_state") else None,
            policy_net=self.policy,
            value_net=self.value,
            sims_per_move=sims_agent,
            root_dirichlet_alpha=None,
            temperature=0.0,
        )

        # Opponent: pure MCTS (no NN priors/value)
        opp_mcts = self.mcts_cls(
            env_cls=self.env_cls,
            encode_state_fn=None,
            policy_net=None,
            value_net=None,
            sims_per_move=sims_opponent,
            root_dirichlet_alpha=None,
            temperature=0.0,
        )

        agent_is_black = (agent_color.lower() == "black")

        while not env.is_terminal():
            player = env.current_player_str()  # 'black' or 'white'
            mcts = agent_mcts if (player == "black") == agent_is_black else opp_mcts
            move = mcts.choose_move(env)  # deterministic argmax over visit counts
            env.step(move)

        result = env.check_result()  # GameResult enum
        agent_won = (result == "BLACK_WIN" and agent_is_black) or (result == "WHITE_WIN" and not agent_is_black)
        return result, agent_won

    def eval_winrate(self, sims_agent, sims_opponent, matches=40, seed_base=42):
        """Alternate colors, record wins."""
        wins = 0
        results = []
        for i in range(matches):
            color = "black" if i % 2 == 0 else "white"
            result, agent_won = self.play_match(sims_agent, sims_opponent, agent_color=color, seed=seed_base + i)
            wins += int(agent_won)
            results.append(result)
        counts = Counter(results)
        return wins / matches, counts