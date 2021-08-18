import sys

from src.snakeAI.agents.dqn.dqn_play import play_dqn
from src.snakeAI.agents.ppo.ppo_play import play_ppo

if __name__ == '__main__':
    args = sys.argv
    if args[1] == "PPO":
        params = {"PATH": args[2], "N_ITERATIONS": int(args[3]), "BOARD_SIZE": eval(args[4]),
                  "PRINT_STATS": bool(args[5]), "HAS_GUI": bool(args[6])}
        play_ppo(**params)

    elif args[1] == "DQN":
        params = {"PATH": args[2], "N_ITERATIONS": int(args[3]), "BOARD_SIZE": eval(args[4]),
                  "PRINT_STATS": bool(args[5]), "HAS_GUI": bool(args[6])}
        play_dqn(**params)

    else:
        sys.exit(-2)
