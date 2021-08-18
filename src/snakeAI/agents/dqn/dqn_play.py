from time import sleep
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def play_dqn(PATH, N_ITERATIONS, BOARD_SIZE=(8, 8), PRINT_STATS=True, HAS_GUI=True):
    agent = Agent(LR=1e-3, N_ACTIONS=3, GAMMA=0.99, BATCH_SIZE=2 ** 6, EPS_START=0.00, EPS_END=0.00, EPS_DEC=0,
                  MAX_MEM_SIZE=2 ** 11)
    agent.load_model(PATH)
    game = SnakeEnv(BOARD_SIZE, HAS_GUI)
    for i in range(1, N_ITERATIONS + 1):
        av, scalar_obs = game.reset()
        scores = 0
        won = False
        while not game.has_ended:
            action = agent.act(av, scalar_obs)

            av_, scalar_obs_, reward, done, won = game.step(action)
            scores += reward

            if game.has_gui:
                game.render()

            av = av_
            scalar_obs = scalar_obs_
            sleep(0.025)
        apple_count = game.apple_count
        if PRINT_STATS:
            print(f"Score: {round(scores, 2)} ||"
                  f" Apple_Counter: {apple_count} ||"
                  f" won: {won} ||"
                  f" epsilon: {agent.epsilon}")
            print("\n")
        sleep(0.025)


if __name__ == '__main__':
    try:
        play_dqn(N_ITERATIONS=1000, PRINT_STATS=True, HAS_GUI=True)
    except KeyboardInterrupt:
        pass
