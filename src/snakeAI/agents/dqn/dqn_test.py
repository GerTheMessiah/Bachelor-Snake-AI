from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, mean
from time import time_ns
from os import environ

from src.common.stop_game_exception import StopGameException
from src.common.utils import save, print_progress, get_random_game_size

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def test_dqn(MODEL_PATH, N_ITERATIONS, BOARD_SIZE=(8, 8), HAS_GUI=False, STATISTIC_RUN_NUMBER=1,
             AGENT_NUMBER=1, RUN_TYPE="baseline", RAND_GAME_SIZE=False, OPTIMIZATION=None, GPU=True):
    try:
        print(fr"{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}\DQN-0{AGENT_NUMBER}-rgs-{RAND_GAME_SIZE}-opt-{OPTIMIZATION}")
        start_time = time_ns()
        agent = Agent(GPU=GPU, EPS_END=0, EPS_DEC=1, EPS_START=0)
        agent.load_model(MODEL_PATH=MODEL_PATH)
        game = SnakeEnv(BOARD_SIZE, HAS_GUI)
        scores, apples, wins, dtime, steps_list, play_ground_size, dq = [], [], [], [], [], [], deque(maxlen=100)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            av, scalar_obs = game.reset(get_random_game_size(i - 1, 1000, 6) if RAND_GAME_SIZE else None)
            while not game.has_ended:
                av, scalar_obs, action = agent.act_test(av, scalar_obs)

                av_, scalar_obs_, reward, done, won = game.step(action, )
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
            play_ground_size.append(game.game.shape)

            t = time_ns()

            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            suffix_3 = f" | eps: {round(agent.EPSILON, 5)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

        save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "test", RUN_TYPE, RAND_GAME_SIZE, agent, dtime, steps_list,
             apples, scores, wins, play_ground_size=play_ground_size, optimization=OPTIMIZATION)

    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {MODEL_DIR_PATH}? y/n \n")
            if answer == 'y':
                save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "test", RUN_TYPE, RAND_GAME_SIZE, agent, dtime,
                     steps_list, apples, scores, wins, play_ground_size=play_ground_size, optimization=OPTIMIZATION)
                repeat = False
            elif answer == 'n':
                repeat = False
            else:
                print("Wrong input!")


if __name__ == '__main__':
    MODEL_PATH = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\optimized-run-02\DQN-02-opt-b-train.model"
    print(MODEL_PATH)
    test_dqn(MODEL_PATH=MODEL_PATH, N_ITERATIONS=5000, HAS_GUI=False, STATISTIC_RUN_NUMBER=2, AGENT_NUMBER=2,
             RUN_TYPE="optimized", RAND_GAME_SIZE=False, OPTIMIZATION="B", GPU=True)

