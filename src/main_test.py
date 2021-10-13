import sys

from src.snakeAI.agents.dqn.dqn_test import test_dqn
from src.snakeAI.agents.ppo.ppo_test import test_ppo

if __name__ == '__main__':
    args = sys.argv
    if args[1] == "PPO":
        params = {"MODEL_PATH": args[2],
                  "N_ITERATIONS": int(args[3]),
                  "BOARD_SIZE": eval(args[4]),
                  "HAS_GUI": args[5],
                  "STATISTIC_RUN_NUMBER": int(args[6]),
                  "AGENT_NUMBER": int(args[7]),
                  "RUN_TYPE": str(args[8]),
                  "RAND_GAME_SIZE": bool(args[9]),
                  "OPTIMIZATION": str(args[10]),
                  "GPU": bool(args[11])}
        test_ppo(**params)

    elif args[1] == "DQN":
        params = {"MODEL_PATH": args[2],
                  "N_ITERATIONS": int(args[3]),
                  "BOARD_SIZE": eval(args[4]),
                  "HAS_GUI": args[5],
                  "STATISTIC_RUN_NUMBER": int(args[6]),
                  "AGENT_NUMBER": int(args[7]),
                  "RUN_TYPE": str(args[8]),
                  "RAND_GAME_SIZE": bool(args[9]),
                  "OPTIMIZATION": str(args[10]),
                  "GPU": bool(args[11])}
        test_dqn(**params)

    else:
        sys.exit(-2)
