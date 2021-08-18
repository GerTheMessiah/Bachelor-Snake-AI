from os import environ
from time import sleep
import torch as T

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def play_ppo(PATH, N_ITERATIONS, BOARD_SIZE=(8, 8), PRINT_STATS=True, HAS_GUI=True):
    agent = Agent()
    try:
        agent.load_model(T.load(PATH))
    except (FileNotFoundError, IOError):
        print("Wrong Path or File doesn't exist!")
        return

    game = SnakeEnv(BOARD_SIZE, HAS_GUI)
    for i in range(1, N_ITERATIONS + 1):
        around_view, cat_obs = game.reset()
        scores = 0
        while not game.has_ended:
            around_view, cat_obs, action, _ = agent.old_policy.act(around_view, cat_obs)
            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new
            sleep(0.255)
        apple_count = game.apple_count
        if PRINT_STATS:
            print(f"Score: {round(scores, 2)} || Apple_Counter: {apple_count} || won: {won}")
            print("\n")
        sleep(0.255)


if __name__ == '__main__':
    play_ppo(100, True)
