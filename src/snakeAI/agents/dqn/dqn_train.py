from statistics import mean
from time import time
import torch as T
from datetime import timedelta
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import plot_learning_curve, file_path, print_progress
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_dqn(n_iterations, print_stats=False, has_gui=False):
    score = 0
    start_time = time()
    scores, apples, wins = [], [], []
    agent = Agent(lr=1e-3, n_actions=3, gamma=0.99, epsilon=1.0, batch_size=2 ** 8, eps_end=0.01, eps_dec=2e-5,
                  max_mem_size=2 ** 13)
    agent.load_model(file_path(dir=r"models\dqn_models", new_save=False, file_name="model"))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=has_gui)
    iter_time = time()
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        while not game.has_ended:
            around_view, cat_obs, action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            score += reward

            agent.mem.add(around_view, cat_obs, action, reward, done, around_view_new, cat_obs_new)
            agent.learn()

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new

        apple_count = game.apple_count
        if print_stats:
            print(f"Score: {round(score, 2)} || Apple_Counter: {apple_count} || won: {won} || epsilon: {agent.epsilon}")
            print("\n")

        scores.append(score)
        wins.append(won)
        apples.append(apple_count)
        score = 0

        t = time()

        time_step = str(timedelta(seconds=t - iter_time) * (n_iterations - i))
        passed_time = str(timedelta(seconds=t - start_time))
        iter_time = time()
        suffix_1 = f"Passed Time: {passed_time} | Remaining Time: {time_step}"
        suffix_2 = f" | Food_avg: {round(mean(apples[-5:]), 2)} | Score_avg: {round(mean(scores[-5:]), 2)}"
        suffix_3 = f" | Wins:{int(mean(wins[-5:]))}"
        print_progress(i, n_iterations, suffix=suffix_1 + suffix_2 + suffix_3)

    path_ = file_path(dir=r"stats\dqn_stats", new_save=True, file_name="stats")
    plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_)
    agent.save_model(new_save=True)


if __name__ == '__main__':
    train_dqn(n_iterations=1000, print_stats=False, has_gui=False)
