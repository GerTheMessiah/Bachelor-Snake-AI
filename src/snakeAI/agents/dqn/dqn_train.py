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


def train_dqn(N_ITERATIONS, LR, GAMMA=0.95, BACH_SIZE=2**6, MAX_MEM_SIZE=2**11, EPS_DEC=1e-5, EPS_END=0.001, BOARD_SIZE=(8, 8)):
    start_time = time_ns()
    scores, apples, wins, dtime, eps, steps_list, dq = [], [], [], [], [], [], deque(maxlen=100)
    agent = Agent(lr=LR, n_actions=3, gamma=GAMMA, batch_size=BACH_SIZE, eps_dec=EPS_DEC, max_mem_size=MAX_MEM_SIZE, eps_end=EPS_END)
    game = SnakeEnv(BOARD_SIZE, False)
    iter_time = time_ns()
    for i in range(1, N_ITERATIONS + 1):
        score = 0
        steps = 0
        around_view, scalar_obs = game.reset()
        while not game.has_ended:
            action = agent.act(around_view, scalar_obs)
            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            score += reward
            steps += 1
            agent.mem.add(around_view, scalar_obs, action, reward, done, around_view_new, cat_obs_new)
            agent.learn()

            if game.has_gui:
                game.render()

            around_view = around_view_new
            scalar_obs = cat_obs_new

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
    try:
        train_dqn(20000, 1e-4, 0.95, 2**6, 2**1, 7.5e-6, 1e-3, (8, 8))
    except KeyboardInterrupt:
        pass
