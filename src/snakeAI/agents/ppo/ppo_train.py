from datetime import timedelta, datetime
from statistics import mean
from time import time
import torch as T
from pandas import DataFrame
from os import environ

from scipy.stats import linregress
from torch.optim.lr_scheduler import ExponentialLR

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import print_progress, file_path, plot_learning_curve
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_play(N_ITERATIONS: int, BOARD_SIZE: tuple):
    try:
        a = 0
        LR_ACTOR, LR_CRITIC, = 0.6e-3, 1.0e-3
        scores, apples, wins, dtime = [], [], [], []
        start_time = time()
        agent = Agent(lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, gamma=0.95, K_epochs=10, eps_clip=0.2, gpu=True)
        #actor_scheduler = ExponentialLR(agent.optimizer, 0.95, verbose=True)
        # try:
        #     agent.load_model(T.load(file_path(dir=r'models\ppo_models', new_save=False, file_name="model")))
        # except FileNotFoundError:
        #     pass
        game = SnakeEnv(BOARD_SIZE, False)
        iter_time = time()
        for i in range(1, N_ITERATIONS + 1):
            score = 0
            av, scalar_obs = game.reset()
            while not game.has_ended:
                av, scalar_obs, action, probs = agent.old_policy.act(av, scalar_obs)

                av_new, scalac_obs_new, reward, done, won = game.step(action)

                agent.mem.store(av, scalar_obs, action, probs, reward, done)
                a += 1
                score += reward

                av = av_new
                scalar_obs = scalac_obs_new

            if len(agent.mem) >= 2 ** 6:
                agent.learn()
                agent.mem.clear_memory()

            scores.append(score)
            apples.append(game.apple_count)
            wins.append(won)
            dtime.append(str(datetime.now()))

            t = time()

            time_step = str(timedelta(seconds=t - iter_time) * (N_ITERATIONS - i))
            passed_time = str(timedelta(seconds=t - start_time))
            iter_time = time()
            suffix_1 = f"Passed Time: {passed_time} | Remaining Time: {time_step}"
            suffix_2 = f" | Food_avg: {round(mean(apples[-5:]), 2)} | Score_avg: {round(mean(scores[-5:]), 2)}"
            suffix_3 = f" | Wins:{int(mean(wins[-5:]))}"
            print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

        path_stat = file_path(dir=r"stats\ppo_stats", new_save=True, file_name="stats")
        plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_stat)
        path_model = file_path(dir=r'models\ppo_models', new_save=False, file_name="model")
        agent.old_policy.load_state_dict(agent.policy.state_dict(destination=None))
        T.save(agent.policy.state_dict(), path_model)
        save_worked = False
        while not save_worked:
            try:
                agent.policy.load_state_dict(T.load(file_path(dir=r"models\ppo_models", new_save=False)))
                save_worked = True
            except FileNotFoundError:
                T.save(agent.policy.state_dict(), path_model)

        columns = ["datetime", "apples", "scores", "wins", "lr_actor", "lr_critic", "board_size_x", "board_size_y"]
        c = {"datetime": dtime, "apples": apples, "scores": scores, "wins": wins,
             "lr_actor": LR_ACTOR, "lr_critic": None, "board_size_x": BOARD_SIZE[0], "board_size_y": BOARD_SIZE[1]}
        df = DataFrame(c, columns=columns)
        df.to_csv(file_path(r'csv\ppo_csv', new_save=True, file_name='train'))
        print(df)
        print(a)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    train_play(60000, (8, 8))
