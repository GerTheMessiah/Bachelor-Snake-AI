import sys
from src.snakeAI.agents.ppo.ppo_train import train_play_ppo
from src.snakeAI.agents.dqn.dqn_train import train_play_dqn

if __name__ == '__main__':
    args = sys.argv
    if args[1] == "PPO":
        params = {"N_ITERATIONS": int(args[2]), "LR_ACTOR": float(args[3]), "LR_CRITIC": float(args[4]),
                  "GAMMA": float(args[5]), "K_EPOCHS": int(args[6]), "EPS_CLIP": float(args[7]),
                  "BOARD_SIZE": eval(args[8]), "PATH": args[9] if args[9] != "" else None,
                  "DO_TOTAL_RUN": bool(args[10]), "GPU": bool(args[11])}
        train_play_ppo(**params)

    elif args[1] == "DQN":
        params = {"N_ITERATIONS": int(args[2]), "LR": float(args[3]), "GAMMA": float(args[4]),
                  "BACH_SIZE": int(args[5]),
                  "MAX_MEM_SIZE": int(args[6]), "EPS_DEC": float(args[7]), "EPS_END": float(args[8]),
                  "BOARD_SIZE": eval(args[9]), "PATH": args[10] if args[10] != "" else None,
                  "DO_TOTAL_RUN": bool(args[11]), "GPU": bool(args[12])}
        train_play_dqn(**params)

    else:
        sys.exit(-2)
