import datetime
from pathlib import Path
from statistics import mean, median
from time import time_ns
from datetime import timedelta
from os import environ
from collections import deque

from pandas import DataFrame

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import plot_learning_curve, print_progress, save_file
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_play_dqn(N_ITERATIONS, LR, GAMMA=0.95, BACH_SIZE=2 ** 6, MAX_MEM_SIZE=2 ** 11, EPS_DEC=1e-5, EPS_END=0.001,
                   BOARD_SIZE=(8, 8), PATH=None, DO_TOTAL_RUN=False, GPU=True):
    params = {"N_ITERATIONS": N_ITERATIONS, "LR": LR, "GAMMA": GAMMA, "BACH_SIZE": BACH_SIZE,
              "MAX_MEM_SIZE": MAX_MEM_SIZE, "EPS_DEC": EPS_DEC, "EPS_END": EPS_END, "BOARD_SIZE": BOARD_SIZE}
    try:
        start_time = time_ns()
        scores, apples, wins, dtime, eps, steps_list, dq = [], [], [], [], [], [], deque(maxlen=100)
        agent = Agent(LR=LR, N_ACTIONS=3, GAMMA=GAMMA, BATCH_SIZE=BACH_SIZE, EPS_DEC=EPS_DEC, MAX_MEM_SIZE=MAX_MEM_SIZE,
                      EPS_END=EPS_END, GPU=GPU)
        game = SnakeEnv(BOARD_SIZE, False)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            steps = 0
            av, scalar_obs = game.reset()
            while not game.has_ended:
                av, scalar_obs, action = agent.act(av, scalar_obs)
                av_, scalar_obs_, reward, done, won = game.step(action)
                score += reward
                steps += 1
                agent.mem.add(av, scalar_obs, action, reward, done, av_, scalar_obs_)
                agent.learn()

                if game.has_gui:
                    game.render()

                av = av_
                scalar_obs = scalar_obs_

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
            suffix_2 = f" | A_avg: {round(mean(apples[-10:]), 2)} | S_avg: {round(mean(scores[-10:]), 2)}"
            suffix_3 = f" | eps: {round(agent.epsilon, 5)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

            if won and not DO_TOTAL_RUN:
                params["N_ITERATIONS"] = i
                params["Time"] = str(dtime[0]).replace(":", "_")
                path_ = save_file(path=PATH, **params)
                plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
                agent.store_model(path=path_)
                columns = ["time", "steps", "apples", "scores", "wins", "epsilon"]
                c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins,
                     "epsilon": eps}
                df = DataFrame(c, columns=columns)
                df.to_csv(path_ + "\\train.csv")
                print(i)
                return

        params["N_ITERATIONS"] = i
        params["Time"] = str(dtime[0]).replace(":", "_")
        path_ = save_file(path=PATH, **params)
        plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
        agent.store_model(path=path_)
        columns = ["time", "steps", "apples", "scores", "wins", "epsilon"]
        c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins, "epsilon": eps}
        df = DataFrame(c, columns=columns)
        df.to_csv(path_ + "\\train.csv")

    except KeyboardInterrupt:
        repeat = True
        path_tmp = str(Path(__file__).parent.parent.parent.parent) + "\\resources\\" if not PATH else PATH
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {path_tmp}? y/n \n")
            if answer == 'y':
                params["N_ITERATIONS"] = i
                params["Time"] = str(dtime[0]).replace(":", "_")
                path_ = save_file(path=PATH, **params)
                plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
                agent.store_model(path=path_)
                columns = ["time", "steps", "apples", "scores", "wins", "epsilon"]
                c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins,
                     "epsilon": eps}
                df = DataFrame(c, columns=columns)
                df.to_csv(path_ + "\\train.csv")
                repeat = False
            elif answer == 'n':
                repeat = False
                pass
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_play_dqn(N_ITERATIONS=30000, LR=2.0e-4, GAMMA=0.95, BACH_SIZE=2**6, MAX_MEM_SIZE=2**11, EPS_DEC=1e-5,
                   EPS_END=0.01, BOARD_SIZE=(8, 8), PATH=None, DO_TOTAL_RUN=False, GPU=True)
