from collections import deque
from datetime import timedelta, datetime
from pathlib import Path
from statistics import mean, median
from time import time_ns
from os import environ

from src.common.stop_game_exception import StopGameException

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.common.utils import print_progress, save
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_ppo(N_ITERATIONS=30000, LR_ACTOR=0.4e-3, LR_CRITIC=1.0e-3, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=1, ALG_TYPE="PPO", AGENT_NUMBER=1, GPU=True):
    try:
        start_time = time_ns()
        scores, apples, wins, dtime, steps_list, dq = [], [], [], [], [], deque(maxlen=100)
        agent = Agent(LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, GAMMA=GAMMA, K_EPOCHS=K_EPOCHS, EPS_CLIP=EPS_CLIP,
                      GPU=GPU)
        game = SnakeEnv(BOARD_SIZE, False)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            av, scalar_obs = game.reset()
            while not game.has_ended:
                av, scalar_obs, action, probs = agent.old_policy.act(av, scalar_obs)

                av_new, scalac_obs_new, reward, done, won = game.step(action)

                agent.mem.store(av, scalar_obs, action, probs, reward, done)
                score += reward

                av = av_new
                scalar_obs = scalac_obs_new

            agent.learn()

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
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2)

            if sum(wins[-100:]) / 100 > 0.6:
                save(ALG_TYPE, AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", agent, dtime, steps_list, apples, scores, wins)
                return

        save(ALG_TYPE, AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", agent, dtime, steps_list, apples, scores, wins)

    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\statistic-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {MODEL_DIR_PATH}? y/n \n")
            if answer == 'y':
                save(ALG_TYPE, AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", agent, dtime, steps_list, apples, scores, wins)
                repeat = False
            elif answer == 'n':
                repeat = False
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_ppo(N_ITERATIONS=30000, LR_ACTOR=0.15e-3, LR_CRITIC=0.3e-3, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=2, ALG_TYPE="PPO", AGENT_NUMBER=3, GPU=True)
