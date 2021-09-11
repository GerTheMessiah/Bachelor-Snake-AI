from collections import deque
from datetime import timedelta, datetime
from pathlib import Path
from statistics import mean, median
from time import time_ns
from os import environ

from scipy.stats import linregress
from torch.optim.lr_scheduler import ExponentialLR

from src.common.stop_game_exception import StopGameException

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.common.utils import print_progress, save, get_random_game_size
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_ppo(N_ITERATIONS=30000, LR_ACTOR=0.4e-3, LR_CRITIC=1.0e-3, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=1, AGENT_NUMBER=1, RUN_TYPE="baseline",
              RAND_GAME_SIZE=False, OPTIMIZATION=None, GPU=True):
    try:
        print(fr"{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}\PPO-0{AGENT_NUMBER}-opt-{OPTIMIZATION.lower()}")
        start_time = time_ns()
        # create date lists for saving episode data
        scores, apples, wins, dtime, steps_list, dq = [], [], [], [], [], deque(maxlen=100)
        # create agent
        agent = Agent(LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, GAMMA=GAMMA, K_EPOCHS=K_EPOCHS, EPS_CLIP=EPS_CLIP,
                      GPU=GPU)
        # p = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\optimized-run-02\PPO-03-opt-b-train.model"
        # print(p)
        # agent.load_model(p)
        # create environment
        game = SnakeEnv(BOARD_SIZE, False)
        # create scheduler for optimization "B"
        scheduler = ExponentialLR(agent.POLICY.OPTIMIZER, 0.95, verbose=True)
        iter_time = time_ns()
        # iterate over 30.000 epochs
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            # get initial observation containing around_view and scalar_obs
            av, scalar_obs = game.reset()
            # repeat until game ends
            while not game.has_ended:
                # determine action
                av, scalar_obs, action, log_probability = agent.OLD_POLICY.act(av, scalar_obs)
                # execute action in environment
                av_, scalac_obs_, reward, is_terminal, won = game.step(action, OPTIMIZATION)
                # update memory
                agent.MEM.store(av, scalar_obs, action, log_probability, reward, is_terminal)
                score += reward
                # set new obs to old
                av = av_
                scalar_obs = scalac_obs_
            # learn
            agent.learn()
            # update data lists
            scores.append(score)
            wins.append(won)
            apples.append(game.apple_count)
            dtime.append(datetime.now().strftime("%H:%M:%S"))
            steps_list.append(game.game.step_counter)

            t = time_ns()
            # print progress for better information
            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2)
            # schedule lr in terms of optimization "B"
            if i > 15000 and i % 100 == 0 and OPTIMIZATION is "B":
                m, b, _, _, _ = linregress(list(range(100)), apples[-100:])
                if m <= 0:
                    scheduler.step()
                    print("\n")
        # save data lists into CSV files.
        save("PPO", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, RAND_GAME_SIZE, agent, dtime, steps_list,
             apples, scores, wins, optimization=OPTIMIZATION)

    # if you want to prematurely stop learning process
    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {MODEL_DIR_PATH}? y/n \n")
            if answer == 'y':
                save("PPO", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, RAND_GAME_SIZE, agent, dtime,
                     steps_list, apples, scores, wins, optimization=OPTIMIZATION)
                repeat = False
            elif answer == 'n':
                repeat = False
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_ppo(N_ITERATIONS=30000, LR_ACTOR=1.0e-4, LR_CRITIC=2.0e-4, GAMMA=0.93, K_EPOCHS=12, EPS_CLIP=0.20,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=2, AGENT_NUMBER=2, RUN_TYPE="optimized", OPTIMIZATION="B",
              GPU=True)
