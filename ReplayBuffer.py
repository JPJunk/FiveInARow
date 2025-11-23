import random
import numpy as np
import torch

from Utils import Transition, BLACK, WHITE

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def get_high_q_transitions(self, q_threshold=0.6):
        return [t for t in self.buffer if t.q_value >= q_threshold]

    def mutate_state(self, state, black_val=BLACK, white_val=WHITE, verbose=False):
        if isinstance(state, torch.Tensor):
            arr = state.cpu().numpy()
        elif isinstance(state, np.ndarray):
            arr = state
        else:
            arr = np.array(state, dtype=np.int8)

        black_count = (arr == black_val).sum()
        white_count = (arr == white_val).sum()

        # Balance counts
        while white_count > black_count:
            coords = np.argwhere(arr == white_val)
            if len(coords) == 0: break
            r, c = coords[random.randrange(len(coords))]
            arr[r, c] = 0
            white_count -= 1

        while black_count > white_count + 1:
            coords = np.argwhere(arr == black_val)
            if len(coords) == 0: break
            r, c = coords[random.randrange(len(coords))]
            arr[r, c] = 0
            black_count -= 1

        if black_count < white_count:
            coords = np.argwhere(arr == white_val)
            if len(coords) > 0:
                r, c = coords[random.randrange(len(coords))]
                arr[r, c] = black_val
        elif black_count - white_count > 1:
            coords = np.argwhere(arr == black_val)
            if len(coords) > 0:
                r, c = coords[random.randrange(len(coords))]
                arr[r, c] = white_val

        if verbose:
            print("mutate_state output:", arr.shape)

        return arr

    @staticmethod
    def mutate_action(state, action, board_size):
        r, c = divmod(action, board_size)
        if state[r, c] != 0:
            empties = np.argwhere(state == 0)
            if len(empties) > 0:
                r, c = empties[random.randint(0, len(empties)-1)]
                action = r * board_size + c
        return action

    def apply_action(self, board, action, player, board_size):
        r, c = divmod(action, board_size)
        next_board = board.copy()
        if next_board[r, c] == 0:  # only place if empty
            next_board[r, c] = player
        return next_board

    def evolve_transitions(self, num_parents=2, num_offspring=100, q_threshold=0.6):
        high_q = self.get_high_q_transitions(q_threshold)
        if len(high_q) < num_parents:
            return

        offspring = []
        for _ in range(num_offspring):
            parents = random.sample(high_q, num_parents)

            combined_state = np.array(parents[0].state, copy=True)
            for p in parents[1:]:
                combined_state = np.where(combined_state != 0, combined_state, p.state)
                # combined_state = np.maximum(combined_state, np.array(p.state)) REMOVES WHITES -1

            avg_q = sum(p.q_value for p in parents) / num_parents


            # print("before mutate_state:", type(combined_state), getattr(combined_state, "shape", None))
            combined_state = self.mutate_state(combined_state)
            # print("after mutate_state:", type(combined_state), getattr(combined_state, "shape", None))

            black_turn = int((combined_state == BLACK).sum())
            white_turn = int((combined_state == WHITE).sum())
            player = WHITE if black_turn > white_turn else BLACK

            if combined_state.ndim == 2:
                board_size = combined_state.shape[0]
            elif combined_state.ndim == 1:
                board_size = int(np.sqrt(combined_state.shape[0]))
            else:
                raise ValueError(f"Unexpected state shape: {combined_state.shape}")

            action = random.randint(0, board_size * board_size - 1)
            action = self.mutate_action(combined_state, action, board_size)

            next_state = self.apply_action(combined_state, action, player, board_size)

            pi = np.zeros(board_size * board_size, dtype=np.float32)
            pi[action] = 1.0
            offspring.append(Transition(
                state=combined_state,
                action=action,
                reward=0.0,
                next_state=next_state,
                done=False,
                player=player,
                q_value=avg_q,
                pi=pi,
                z=None,  # fill with Gomoku(next_state).check_result() if desired
                meta={'type': 'evolved'}
            ))

        for t in offspring:
            self.push(t)
