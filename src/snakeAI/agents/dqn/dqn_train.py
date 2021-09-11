import datetime
from pathlib import Path
from statistics import mean, median
from time import time_ns
from datetime import timedelta
from os import environ
from collections import deque

from scipy.stats import linregress
from torch.optim.lr_scheduler import ExponentialLR

from src.common.stop_game_exception import StopGameException

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.common.utils import print_progress, save, get_random_game_size
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_dqn(N_ITERATIONS, LR, GAMMA=0.95, BACH_SIZE=2 ** 6, MAX_MEM_SIZE=2 ** 11, EPS_DEC=1e-5, EPS_END=0.001,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=1, AGENT_NUMBER=1, RUN_TYPE="baseline", RAND_GAME_SIZE=False,
              OPTIMIZATION=None, GPU=False):
    try:
        print(fr"{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}\DQN-0{AGENT_NUMBER}-rgs{RAND_GAME_SIZE}-opt-{OPTIMIZATION}")
        start_time = time_ns()
        scores, apples, wins, dtime, eps, steps_list, dq = [], [], [], [], [], [], deque(maxlen=100)
        agent = Agent(LR=LR, N_ACTIONS=3, GAMMA=GAMMA, BATCH_SIZE=BACH_SIZE, EPS_DEC=EPS_DEC, MAX_MEM_SIZE=MAX_MEM_SIZE,
                      EPS_END=EPS_END, GPU=GPU, EPS_START=1.0)
        # agent.load_model(r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\baseline-run-02\DQN-02-train.model")
        game = SnakeEnv(BOARD_SIZE, False)
        scheduler = ExponentialLR(agent.Q_NET.OPTIMIZER, 0.95, verbose=True)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            av, scalar_obs = game.reset(get_random_game_size(i - 1) if RAND_GAME_SIZE else None)
            while not game.has_ended:
                av, scalar_obs, action = agent.act(av, scalar_obs)
                av_, scalar_obs_, reward, done, won = game.step(action, reward_function=OPTIMIZATION)
                score += reward
                agent.MEM.store(av, scalar_obs, action, reward, done, av_, scalar_obs_)
                if game.game.step_counter % 5 == 0:
                    agent.learn()

                av = av_
                scalar_obs = scalar_obs_

            scores.append(score)
            wins.append(won)
            apples.append(game.apple_count)
            dtime.append(datetime.datetime.now().strftime("%H:%M:%S"))
            eps.append(agent.EPSILON)
            steps_list.append(game.game.step_counter)

            t = time_ns()
            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            suffix_3 = f" | eps: {round(agent.EPSILON, 5)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

            if won:
                save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, RAND_GAME_SIZE, agent, dtime,
                     steps_list, apples, scores, wins, optimization=OPTIMIZATION)
                return

            if i > 10_000 and i % 100 == 0 and OPTIMIZATION is "B":
                m, b, _, _, _ = linregress(list(range(100)), apples[-100:])
                if m <= 0:
                    scheduler.step()
                    print("\n")

        save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, RAND_GAME_SIZE, agent, dtime,
             steps_list, apples, scores, wins, optimization=OPTIMIZATION)

    except (KeyboardInterrupt, StopGameException):
        repeat = True
        path_tmp = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {path_tmp}? y/n \n")
            if answer == 'y':
                save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, RAND_GAME_SIZE, agent, dtime,
                     steps_list, apples, scores, wins, optimization=OPTIMIZATION)
                repeat = False
            elif answer == 'n':
                repeat = False
                pass
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_dqn(N_ITERATIONS=30_000, LR=2.0e-4, GAMMA=0.90, BACH_SIZE=2 ** 7, MAX_MEM_SIZE=2 ** 12, EPS_DEC=6.0e-5,
              EPS_END=0.005, BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=2, AGENT_NUMBER=2, RUN_TYPE="optimized",
              RAND_GAME_SIZE=False, OPTIMIZATION="B", GPU=True)
