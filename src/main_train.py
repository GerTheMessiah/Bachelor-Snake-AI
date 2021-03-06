import sys
from src.snakeAI.agents.ppo.ppo_train import train_ppo
from src.snakeAI.agents.dqn.dqn_train import train_dqn

if __name__ == '__main__':
    try:
        args = sys.argv
        if args[1] == "PPO":
            params = {"N_ITERATIONS": int(args[2]), "LR_ACTOR": float(args[3]), "LR_CRITIC": float(args[4]),
                      "GAMMA": float(args[5]), "K_EPOCHS": int(args[6]), "EPS_CLIP": float(args[7]),
                      "BOARD_SIZE": eval(args[8]), "STATISTIC_RUN_NUMBER": int(args[9]),
                      "AGENT_NUMBER": int(args[10]), "RUN_TYPE": str(args[11]), "OPTIMIZATION": str(args[12]),
                      "GPU": bool(args[13])}
            train_ppo(**params)

        elif args[1] == "DQN":
            params = {"N_ITERATIONS": int(args[2]), "LR": float(args[3]), "GAMMA": float(args[4]),
                      "BACH_SIZE": int(args[5]), "MAX_MEM_SIZE": int(args[6]), "EPS_DEC": float(args[7]),
                      "EPS_END": float(args[8]), "BOARD_SIZE": eval(args[9]), "STATISTIC_RUN_NUMBER": int(args[10]),
                      "AGENT_NUMBER": int(args[11]), "RUN_TYPE": str(args[12]), "OPTIMIZATION": str(args[13]),
                      "GPU": bool(args[14])}
            train_dqn(**params)

        else:
            print("\nWrong Parameters")
            sys.exit(-2)
    except Exception:
        print(print("\nWrong Input."))
