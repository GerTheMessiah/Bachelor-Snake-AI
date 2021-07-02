import datetime
from statistics import mean, median
from time import time_ns
from datetime import timedelta
from os import environ
from collections import deque

from pandas import DataFrame

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import plot_learning_curve, file_path, print_progress
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_dqn(N_ITERATIONS, has_gui=False):
    LR, BOARD_SIZE = 5.0e-4, (8, 8)
    start_time = time_ns()
    scores, apples, wins, dtime, eps, steps_list, dq = [], [], [], [], [], [], deque(maxlen=100)
    agent = Agent(lr=LR, n_actions=3, batch_size=2 ** 6, eps_dec=5e-6, max_mem_size=2 ** 12)
    agent.load_model()
    game = SnakeEnv()
    game.post_init(field_size=BOARD_SIZE, has_gui=has_gui)
    iter_time = time_ns()
    for i in range(1, N_ITERATIONS + 1):
        score = 0
        steps = 0
        around_view, cat_obs = game.reset()
        while not game.has_ended:
            around_view, cat_obs, action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            score += reward
            steps += 1
            agent.mem.add(around_view, cat_obs, action, reward, done, around_view_new, cat_obs_new)
            agent.learn()

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new

        apple_count = game.apple_count

        scores.append(score)
        wins.append(won)
        apples.append(apple_count)
        dtime.append(datetime.datetime.now().strftime("%H:%M:%S"))
        eps.append(agent.epsilon)
        steps_list.append(steps)

        t = time_ns()
        dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
        time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
        passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
        iter_time = time_ns()
        suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
        suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
        suffix_3 = f" | eps: {round(agent.epsilon, 5)}"
        print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

    path_ = file_path(dir=r"stats\dqn_stats", new_save=True, file_name="stats")
    plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + ".png")
    agent.save_model(new_save=True)
    columns = ["time", "steps", "apples", "scores", "wins", "epsilon"]
    c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins, "epsilon": eps}
    df = DataFrame(c, columns=columns)
    df.to_csv(file_path(r'csv\dqn_csv', new_save=True, file_name='train') + ".csv")


if __name__ == '__main__':
    train_dqn(N_ITERATIONS=30000, has_gui=False)
