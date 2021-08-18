from collections import deque
from datetime import timedelta, datetime
from pathlib import Path
from statistics import mean, median
from time import time_ns
from pandas import DataFrame
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import plot_learning_curve, save_file, print_progress
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_play_ppo(N_ITERATIONS=20000, LR_ACTOR=0.4e-3, LR_CRITIC=1.0e-3, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2,
                   BOARD_SIZE=(8, 8), PATH=None, DO_TOTAL_RUN=False, GPU=True):
    params = {"N_ITERATIONS": N_ITERATIONS, "LR_ACTOR": LR_ACTOR, "LR_CRITIC": LR_CRITIC, "GAMMA": GAMMA,
              "K_EPOCHS": K_EPOCHS, "EPS_CLIP": EPS_CLIP, "BOARD_SIZE": BOARD_SIZE}
    try:
        start_time = time_ns()
        scores, apples, wins, dtime, steps_list, dq = [], [], [], [], [], deque(maxlen=100)
        agent = Agent(LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, GAMMA=GAMMA, K_EPOCHS=K_EPOCHS, EPS_CLIP=EPS_CLIP,
                      GPU=GPU)
        game = SnakeEnv(BOARD_SIZE, False)
        iter_time = time_ns()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            steps = 0
            av, scalar_obs = game.reset()
            while not game.has_ended:
                av, scalar_obs, action, probs = agent.old_policy.act(av, scalar_obs)

                av_new, scalac_obs_new, reward, done, won = game.step(action)

                agent.mem.store(av, scalar_obs, action, probs, reward, done)
                steps += 1
                score += reward

                av = av_new
                scalar_obs = scalac_obs_new

            agent.learn()

            scores.append(score)
            wins.append(won)
            apples.append(game.apple_count)
            dtime.append(datetime.now().strftime("%H:%M:%S"))
            steps_list.append(steps)

            t = time_ns()

            dq.append(((t - iter_time) / 1_000) * (N_ITERATIONS - i))
            time_step = str(timedelta(microseconds=(median(dq)))).split('.')[0]
            passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
            iter_time = time_ns()
            suffix_1 = f"P.Time: {passed_time} | R.Time: {time_step}"
            suffix_2 = f" | A_avg: {round(mean(apples[-10:]), 2)} | S_avg: {round(mean(scores[-10:]), 2)}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2)

            if sum(wins[-10:]) > 5 and not DO_TOTAL_RUN:
                params["N_ITERATIONS"] = i
                params["Time"] = str(dtime[0]).replace(":", "_")
                path_ = save_file(**params)
                plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
                agent.store_model(path=path_)
                columns = ["time", "steps", "apples", "scores", "wins"]
                c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins}
                df = DataFrame(c, columns=columns)
                df.to_csv(path_ + "\\train.csv")
                print(i)
                return

        params["N_ITERATIONS"] = i
        params["Time"] = str(dtime[0]).replace(":", "_")
        path_ = save_file(**params)
        plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
        agent.store_model(path=path_)
        columns = ["time", "steps", "apples", "scores", "wins"]
        c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins}
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
                path_ = save_file(**params)
                plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_ + "\\stats.png")
                agent.store_model(path=path_)
                columns = ["time", "steps", "apples", "scores", "wins"]
                c = {"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins}
                df = DataFrame(c, columns=columns)
                df.to_csv(path_ + "\\train.csv")
                repeat = False
            elif answer == 'n':
                repeat = False
                pass
            else:
                print("Wrong input!")


if __name__ == '__main__':
    train_play_ppo(N_ITERATIONS=30000, LR_ACTOR=0.15e-3, LR_CRITIC=0.4e-3, GAMMA=0.99, K_EPOCHS=10, EPS_CLIP=0.2,
                   BOARD_SIZE=(8, 8), PATH=None, DO_TOTAL_RUN=False, GPU=True)
