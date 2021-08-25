import sys

from src.snakeAI.agents.dqn.dqn_test import test_dqn
from src.snakeAI.agents.ppo.ppo_test import test_ppo

if __name__ == '__main__':
    args = sys.argv
    if args[1] == "PPO":
        params = {"MODEL_PATH": args[2], "N_ITERATIONS": int(args[3]), "BOARD_SIZE": eval(args[4]),
                  "STATISTIC_RUN_NUMBER": int(args[5]), "ALG_TYPE": "PPO", "AGENT_NUMBER": int(args[6]),
                  "GPU": bool(args[7])}
        test_ppo(**params)

    elif args[1] == "DQN":
        params = {"MODEL_PATH": args[2], "N_ITERATIONS": int(args[3]), "BOARD_SIZE": eval(args[4]),
                  "STATISTIC_RUN_NUMBER": int(args[5]), "ALG_TYPE": "DQN", "AGENT_NUMBER": int(args[6]),
                  "GPU": bool(args[7])}
        test_dqn(**params)

    else:
        sys.exit(-2)
