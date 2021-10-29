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

from src.common.utils import print_progress, save
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv

"""
This method is the central training routine of the PPO agent.
@:param N_ITERATIONS: Iteration which should be processed for learning.
@:param LR_ACTOR: Learning rate of the actor.
@:param LR_CRITIC: Learning rate of the critic.
@:param GAMMA: Discounting factor. Important for the time preference.
@:param K_EPOCHS: Number of the epochs for learning with one data batch.
@:param EPS_CLIP: Clip Value of the PPO
@:param BOARD_SIZE: Size of the playground. E.g. (8, 8)
@:param STATISTIC_RUN_NUMBER: Number of the Run. Important for saving the generated data.
@:param AGENT_NUMBER: Number of the to be examined agent. Important for saving.
@:param RUN_TYPE: Type of the statistic run. BaseLine or Optimized.
@:param OPTIMIZATION: "A", "B", "AB", "None"
@:param GPU: Should the GPU be used.
"""
def train_ppo(N_ITERATIONS=30000, LR_ACTOR=0.4e-3, LR_CRITIC=1.0e-3, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=1, AGENT_NUMBER=1, RUN_TYPE="baseline",
              OPTIMIZATION=None, GPU=True):
    try:
        print(fr"{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}\PPO-0{AGENT_NUMBER}-opt-{str(OPTIMIZATION).lower()}")
        start_time = time_ns()
        # Initialize data lists.
        scores, apples, wins, dtime, steps_list, dq = [], [], [], [], [], deque(maxlen=100)
        # Initialize agent.
        agent = Agent(LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, GAMMA=GAMMA, K_EPOCHS=K_EPOCHS, EPS_CLIP=EPS_CLIP,
                      GPU=GPU)
        # Initialize environment.
        game = SnakeEnv(BOARD_SIZE, False)
        # Initialize scheduler of the lr. Optimization "B".
        scheduler = ExponentialLR(agent.POLICY.OPTIMIZER, 0.95, verbose=True)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            # Game reset for getting the initial observation consisting of around_view and scalar_obs.
            av, scalar_obs = game.reset()
            # Until game ended.
            while not game.has_ended:
                # Determine action.
                av, scalar_obs, action, log_probability = agent.OLD_POLICY.act(av, scalar_obs)
                # Process action in the environment and get new around_view (av_) and scalar_obs (so_).
                av_, scalac_obs_, reward, is_terminal, won = game.step(action, OPTIMIZATION)
                # Store experiences into memory.
                agent.MEM.store(av, scalar_obs, action, log_probability, reward, is_terminal)
                score += reward
                # New obs resets old obs.
                av = av_
                scalar_obs = scalac_obs_
            # learn
            agent.learn()
            # Insert game data into data lists.
            scores.append(score)
            wins.append(won)
            apples.append(game.apple_count)
            dtime.append(datetime.now().strftime("%H:%M:%S"))
            steps_list.append(game.game.step_counter)

            t = time_ns()
            # Print progress of testing.
            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-5:]), 2)} | S_avg: {round(mean(scores[-5:]), 2)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2)
            # If optimization "B" is active then schedule lr.
            if i > 15000 and i % 100 == 0 and "B" in OPTIMIZATION:
                m, b, _, _, _ = linregress(list(range(100)), apples[-100:])
                if m <= 0:
                    scheduler.step()
                    print("\n")
        # Save data of data lists.
        save("PPO", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, False, agent, dtime, steps_list,
             apples, scores, wins, optimization=OPTIMIZATION)

    # If you want to prematurely stop learning process.
    except (KeyboardInterrupt, StopGameException):
        repeat = True
        MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + f"\\resources\\{RUN_TYPE}-run-0{STATISTIC_RUN_NUMBER}"
        while repeat:
            answer = input(f"\nDo you want to save the files in a new Folder at {MODEL_DIR_PATH}? y/n \n")
            if answer == 'y':
                save("PPO", AGENT_NUMBER, STATISTIC_RUN_NUMBER, "train", RUN_TYPE, False, agent, dtime,
                     steps_list, apples, scores, wins, optimization=OPTIMIZATION)
                repeat = False
            elif answer == 'n':
                repeat = False
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_ppo(N_ITERATIONS=30000, LR_ACTOR=1.5e-4, LR_CRITIC=3.0e-4, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.20,
              BOARD_SIZE=(8, 8), STATISTIC_RUN_NUMBER=1, AGENT_NUMBER=3, RUN_TYPE="optimized", OPTIMIZATION="AB",
              GPU=True)
