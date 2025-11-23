import torch
import traceback

from Utils import DEVICE

from PolicyAndValueNets import PolicyNet, ValueNet

import os
os.chdir(r"C:\Users\XXX\Documents\PythonProjects\Gomoku") # Change to project directory

# # Registry of available network classes
NETWORK_REGISTRY = {
    # "CompactPolicyCNN": CompactPolicyCNN,
    # "CompactValueCNN": CompactValueCNN,
    "PolicyNet": PolicyNet,
    "ValueNet": ValueNet,
}

class AgentPersistence:
    """Helper for saving/loading DRL agent state."""
    @staticmethod
    def load(agent, filename="gomoku_agent.pth", load_replay=True, board_size=15):
        try:
            print(os.path.exists(filename), os.path.getsize(filename))
            print(os.path.abspath(filename))
            print(f"Loading on device: {DEVICE}")            
            checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False) # , weights_only=False
            print(checkpoint.keys())
        except Exception as e:
            print(f"[Persistence] Failed to load {filename}: {type(e).__name__} - {e}")
            traceback.print_exc()
            return

        board_size = checkpoint.get("board_size", board_size)
        model_type = checkpoint.get("model_type", {})

        # Instantiate correct classes if possible
        if model_type:
            policy_cls = NETWORK_REGISTRY.get(model_type.get("policy"))
            value_cls  = NETWORK_REGISTRY.get(model_type.get("value"))
            if policy_cls and value_cls:
                agent.policy = policy_cls(board_size=board_size).to(DEVICE)
                agent.value  = value_cls(board_size=board_size).to(DEVICE)
                agent.policy_opt = torch.optim.Adam(agent.policy.parameters(), lr=5e-4, weight_decay=1e-4)
                agent.value_opt  = torch.optim.Adam(agent.value.parameters(), lr=5e-4, weight_decay=1e-4)
                print(f"[Persistence] Instantiated {model_type['policy']}/{model_type['value']} before loading.")
            else:
                print(f"[Persistence] Warning: Unknown model types {model_type}. Using existing agent networks.")

        # Load weights
        if "policy" in checkpoint:
            agent.policy.load_state_dict(checkpoint["policy"])
        if "value" in checkpoint:
            agent.value.load_state_dict(checkpoint["value"])

        # Load optimizer states
        if "policy_opt" in checkpoint:
            agent.policy_opt.load_state_dict(checkpoint["policy_opt"])
        if "value_opt" in checkpoint:
            agent.value_opt.load_state_dict(checkpoint["value_opt"])

        # Metadata
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.epsilon_step = checkpoint.get("epsilon_step", getattr(agent, "epsilon_step", 0))
        agent.game_counter = checkpoint.get("game_counter", getattr(agent, "game_counter", 0))

        # Replay buffer
        if load_replay and "replay" in checkpoint:
            agent.replay.buffer = checkpoint["replay"]
            agent.replay.idx = checkpoint.get("replay_idx", len(agent.replay.buffer) % agent.replay.capacity)

        print(f"[Persistence] Agent loaded ({model_type.get('policy')}/{model_type.get('value')}) from {filename}"
            if model_type else f"[Persistence] Agent loaded from {filename} (model type not recorded)")

    @staticmethod
    def save(agent, filename="gomoku_agent.pth", save_replay=True):
        checkpoint = {
            "policy": agent.policy.state_dict(),
            "value": agent.value.state_dict(),
            "policy_opt": agent.policy_opt.state_dict(),
            "value_opt": agent.value_opt.state_dict(),
            "epsilon": agent.epsilon,
            "epsilon_step": getattr(agent, "epsilon_step", 0),
            "game_counter": getattr(agent, "game_counter", 0),
            "board_size": getattr(agent.policy, "board_size", 15),
            "model_type": {
                "policy": agent.policy.__class__.__name__,
                "value": agent.value.__class__.__name__,
            },
        }
        if save_replay:
            checkpoint["replay"] = agent.replay.buffer
            checkpoint["replay_idx"] = agent.replay.idx

        torch.save(checkpoint, filename)
        print(f"[Persistence] Agent ({checkpoint['model_type']['policy']}/"
              f"{checkpoint['model_type']['value']}) saved to {filename}")
        
        print(f"[Persistence] Saved with replay size={len(agent.replay)}, epsilon={agent.epsilon:.4f}")

    # @staticmethod
    # def load(agent, filename="gomoku_agent.pth", load_replay=True, board_size=15):

    #     try:
    #         # print(os.path.exists(filename), os.path.getsize(filename))
    #         # checkpoint = torch.load(filename, map_location="cpu")
    #         # print(checkpoint.keys())
    #         # print(f"Loading on device: {DEVICE}")
    #         checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False)
    #         # checkpoint = torch.load(filename, map_location=DEVICE)
    #     except Exception as e:
    #         print(f"[Persistence] Failed to load {filename}: {type(e).__name__} - {e}")
    #         traceback.print_exc()
    #         return

    #     board_size = checkpoint.get("board_size", board_size)
    #     model_type = checkpoint.get("model_type", {})

    #     if model_type:
    #         policy_cls = NETWORK_REGISTRY.get(model_type.get("policy"))
    #         value_cls  = NETWORK_REGISTRY.get(model_type.get("value"))
    #         if policy_cls and value_cls:
    #             agent.policy = policy_cls(board_size=board_size).to(DEVICE)
    #             agent.value  = value_cls(board_size=board_size).to(DEVICE)
    #             agent.policy_opt = torch.optim.Adam(agent.policy.parameters(), lr=5e-4, weight_decay=1e-4)
    #             agent.value_opt  = torch.optim.Adam(agent.value.parameters(), lr=5e-4, weight_decay=1e-4)
    #             print(f"[Persistence] Instantiated {model_type['policy']}/{model_type['value']} before loading.")
    #         else:
    #             print(f"[Persistence] Warning: Unknown model types {model_type}. Using existing agent networks.")

    #     agent.policy.load_state_dict(checkpoint["policy"])
    #     agent.value.load_state_dict(checkpoint["value"])

    #     if "policy_opt" in checkpoint:
    #         agent.policy_opt.load_state_dict(checkpoint["policy_opt"])
    #     if "value_opt" in checkpoint:
    #         agent.value_opt.load_state_dict(checkpoint["value_opt"])

    #     agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
    #     agent.epsilon_step = checkpoint.get("epsilon_step", getattr(agent, "epsilon_step", 0))
    #     agent.game_counter = checkpoint.get("game_counter", getattr(agent, "game_counter", 0))

    #     if load_replay and "replay" in checkpoint:
    #         agent.replay.buffer = checkpoint["replay"]
    #         agent.replay.idx = checkpoint.get("replay_idx", len(agent.replay.buffer) % agent.replay.capacity)

    #     if model_type:
    #         print(f"[Persistence] Agent loaded ({model_type.get('policy')}/{model_type.get('value')}) from {filename}")
    #     else:
    #         print(f"[Persistence] Agent loaded from {filename} (model type not recorded)")
