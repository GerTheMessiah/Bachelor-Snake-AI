from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, mean
from time import sleep, time_ns
from os import environ
import torch as T

from src.common.stop_game_exception import StopGameException
from src.common.utils import save, print_progress

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def test_dqn(MODEL_PATH, N_ITERATIONS, BOARD_SIZE=(8, 8), HAS_GUI=False, STATISTIC_RUN_NUMBER=1, ALG_TYPE="DQN",
             AGENT_NUMBER=1, GPU=True):
    try:
        start_time = time_ns()
        agent = Agent(LR=1e-3, N_ACTIONS=3, GAMMA=0.99, BATCH_SIZE=2 ** 6, EPS_START=0.00, EPS_END=0.00, EPS_DEC=0,
                      MAX_MEM_SIZE=2 ** 11, GPU=GPU)
        agent.load_model(MODEL_PATH=MODEL_PATH)
        game = SnakeEnv(BOARD_SIZE, HAS_GUI)
        scores, apples, wins, dtime, steps_list, dq = [], [], [], [], [], deque(maxlen=100)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            av, scalar_obs = game.reset()
            score = 0
            while not game.has_ended:
                action = agent.act_test(av, scalar_obs)

                av_, scalar_obs_, reward, done, won = game.step(action)
                score += reward

                if game.has_gui:
                    game.render()

                av = av_
                scalar_obs = scalar_obs_

                scores.append(score)
                wins.append(won)
                apples.append(game.apple_count)
                dtime.append(datetime.now().strftime("%H:%M:%S"))
                steps_list.append(game.game.step_counter)

                t = time_ns()

                dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
                time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
                passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
                iter_time = time_ns()
                suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
                suffix_2 = f" | A_avg: {round(mean(apples[-10:]), 2)} | S_avg: {round(mean(scores[-10:]), 2)}"
                suffix_3 = f" | eps: {round(agent.epsilon, 5)}"
                print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

            save(ALG_TYPE, AGENT_NUMBER, STATISTIC_RUN_NUMBER, "test", agent, dtime, steps_list, apples, scores, wins)

    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\statistic-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {MODEL_DIR_PATH}? y/n \n")
            if answer == 'y':
                save(ALG_TYPE, AGENT_NUMBER, STATISTIC_RUN_NUMBER, "test", agent, dtime, steps_list, apples, scores, wins)
                repeat = False
            elif answer == 'n':
                repeat = False
            else:
                print("Wrong input!")


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\statistic-run-01\DQN-01-train.model"
    test_dqn(MODEL_PATH=MODEL_PATH, N_ITERATIONS=30000, HAS_GUI=True, STATISTIC_RUN_NUMBER=1, ALG_TYPE="DQN",
             AGENT_NUMBER=1, GPU=True)

