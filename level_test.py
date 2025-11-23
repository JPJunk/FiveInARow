
class TrainingStage:
    def __init__(self, name, sims_per_move, lr, augmentation_phase):
        self.name = name
        self.sims_per_move = sims_per_move
        self.lr = lr
        self.augmentation_phase = augmentation_phase  # e.g., 1: rotations, 2: +flip_ud, etc.

class LevelTestScheduler:
    def __init__(self, agent, eval_harness, stages, eval_every_games=100):
        self.agent = agent
        self.eval = eval_harness
        self.stages = stages
        self.current_idx = 0
        self.eval_every_games = eval_every_games

    def maybe_eval_and_promote(self):
        if self.agent.game_counter % self.eval_every_games != 0:
            return False

        # Baseline tiers
        tiers = [
            ("weak", 100, 40, 0.65),
            ("peer", self.stages[self.current_idx].sims_per_move, 60, 0.55),
            ("strong", max(400, self.stages[self.current_idx].sims_per_move * 2), 80, 0.45),
        ]

        sims_agent = self.stages[self.current_idx].sims_per_move
        all_pass = True

        for name, sims_opp, matches, threshold in tiers:
            wr, counts = self.eval.eval_winrate(sims_agent=sims_agent, sims_opponent=sims_opp, matches=matches)
            print(f"[LevelTest] Stage={self.stages[self.current_idx].name} vs {name}({sims_opp}) "
                  f"winrate={wr:.3f} counts={dict(counts)} threshold={threshold:.2f}")
            if wr < threshold:
                all_pass = False

        if all_pass and self.current_idx + 1 < len(self.stages):
            self.promote_to_next_stage()
            return True
        return False

    def promote_to_next_stage(self):
        prev = self.stages[self.current_idx]
        self.current_idx += 1
        nxt = self.stages[self.current_idx]

        # Save checkpoint before promotion
        self.agent.save_checkpoint(tag=f"pre_promotion_{prev.name}")

        # Apply new params: sims_per_move, LR, augmentation
        self.agent.config["sims_per_move"] = nxt.sims_per_move
        for g in self.agent.policy_opt.param_groups:
            g["lr"] = nxt.lr
        for g in self.agent.value_opt.param_groups:
            g["lr"] = nxt.lr
        self.agent.augmentation_phase = nxt.augmentation_phase

        # Persist
        self.agent.save_checkpoint(tag=f"post_promotion_{nxt.name}")
        print(f"[Promotion] {prev.name} -> {nxt.name} | sims={nxt.sims_per_move} lr={nxt.lr} aug_phase={nxt.augmentation_phase}")