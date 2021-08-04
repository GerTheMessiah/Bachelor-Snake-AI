from time import sleep
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from src.snakeAI.agents.common.utils import file_path
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def play_dqn(n_iterations, print_stats=True, has_gui=True):
    BOARD_SIZE = (8, 8)
    agent = Agent(lr=1e-3, n_actions=3, gamma=0.99, epsilon_start=0.0, batch_size=2 ** 8, eps_end=0.00, eps_dec=0,
                  max_mem_size=2 ** 13)
    agent.load_model(file_path(dir=r"models\dqn_models", new_save=False, file_name="model"))
    game = SnakeEnv(BOARD_SIZE, has_gui)
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        scores = 0
        won = False
        while not game.has_ended:
            action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new
            sleep(0.025)
        apple_count = game.apple_count
        if print_stats:
            print(f"Score: {round(scores, 2)} ||"
                  f" Apple_Counter: {apple_count} ||"
                  f" won: {won} ||"
                  f" epsilon: {agent.epsilon}")
            print("\n")
        sleep(0.025)


if __name__ == '__main__':
    try:
        play_dqn(1000, print_stats=True, has_gui=True)
    except KeyboardInterrupt:
        pass
