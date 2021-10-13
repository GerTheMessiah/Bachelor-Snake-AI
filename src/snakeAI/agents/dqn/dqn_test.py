from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, mean
from time import time_ns, sleep
from os import environ

from src.common.stop_game_exception import StopGameException
from src.common.utils import save, print_progress, get_random_game_size

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv

"""
Test routine for generating the test data.
@:param MODEL_PATH: Path of the model to be loaded.
@:param N_ITERATIONS: Iterations to be done (Number of games to be played)
@:param BOARD_SIZE: Size of the playground.
@:param HAS_GUI: If true gui will be started else gui will be deactivated.
@:param STATISTIC_RUN_NUMBER: Number of the statistic run. Important for the saving path.
@:param AGENT_NUMBER: Number of the agent. Important for the differentiating the agents.
@:param RUN_TYPE: What is the type of the run? (baseline or optimized)
@:param RAND_GAME_SIZE: If ture, the playground size will change randomly from game to game.
@:param OPTIMIZATION: Which optimization should be used? ("A", "B", None)
@:param GPU: Should the GPU be used for training? (Only available for NVIDIA GPUs.
"""
def test_dqn(MODEL_PATH, N_ITERATIONS, BOARD_SIZE=(8, 8), HAS_GUI=False, STATISTIC_RUN_NUMBER=1,
             AGENT_NUMBER=1, RUN_TYPE="baseline", RAND_GAME_SIZE=False, OPTIMIZATION=None, GPU=True):
    try:
        print(fr"{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}\DQN-0{AGENT_NUMBER}-rgs-{RAND_GAME_SIZE}-opt-{OPTIMIZATION}")
        start_time = time_ns()
        # Generate agent.
        agent = Agent(GPU=GPU, EPS_END=0, EPS_DEC=1, EPS_START=0)
        agent.load_model(PATH=MODEL_PATH)
        # Generate game.
        game = SnakeEnv(BOARD_SIZE, HAS_GUI)
        # Initialize data lists.
        scores, apples, wins, dtime, steps_list, play_ground_size, dq = [], [], [], [], [], [], deque(maxlen=100)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            # Game reset for getting the initial observation consisting of around_view and scalar_obs.
            av, scalar_obs = game.reset(get_random_game_size(i - 1, 1000, 6) if RAND_GAME_SIZE else None)
            # Until game ended.
            while not game.has_ended:
                # Determine action.
                av, scalar_obs, action = agent.act_test(av, scalar_obs)
                # Process action in the environment and get new around_view (av_) and scalar_obs (so_)
                av_, scalar_obs_, reward, done, won = game.step(action)
                score += reward

                if game.has_gui:
                    game.render()
                    sleep(0.1)
                # New obs resets old obs
                av = av_
                scalar_obs = scalar_obs_
            # Insert game data into data lists.
            scores.append(score)
            wins.append(won)
            apples.append(game.apple_count)
            dtime.append(datetime.now().strftime("%H:%M:%S"))
            steps_list.append(game.game.step_counter)
            play_ground_size.append(game.game.shape)

            t = time_ns()
            # Print progress of testing.
            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            suffix_3 = f" | eps: {round(agent.EPSILON, 5)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)
            if HAS_GUI:
                sleep(0.1)
        # Save data of data lists.
        save("DQN", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "test", RUN_TYPE, RAND_GAME_SIZE, agent, dtime, steps_list,
             apples, scores, wins, play_ground_size=play_ground_size, optimization=OPTIMIZATION)

    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(
            Path(__file__).parent.parent.parent.parent) + f"\\resources\\{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}"
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
    MODEL_PATH = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\baseline-run-01\DQN-03-train.model"
    test_dqn(MODEL_PATH=MODEL_PATH, N_ITERATIONS=5000, HAS_GUI=False, STATISTIC_RUN_NUMBER=1, AGENT_NUMBER=3,
             RUN_TYPE="baseline", RAND_GAME_SIZE=False, OPTIMIZATION="", GPU=True)
