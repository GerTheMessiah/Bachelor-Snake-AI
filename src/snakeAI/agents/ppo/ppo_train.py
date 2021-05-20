import time
import datetime
from statistics import mean
import torch as T
from scipy.stats import linregress
from torch.optim.lr_scheduler import ExponentialLR

from os import environ
import ray

from src.snakeAI.agents.ppo.memoryPPO import Memory
from src.snakeAI.agents.ppo.actor_critic import ActorCritic
from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.agents.common.utils import print_progress, plot_learning_curve, file_path

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from src.snakeAI.gym_game.snake_env.snake_env import SnakeEnv


@ray.remote
class ParameterServer:
    def __init__(self):
        self.params = ActorCritic(n_actions=3)
        self.load_net()

    def get_params(self):
        return {k: v.cpu() for k, v in self.params.state_dict().items()}

    def update_params(self, state_dict):
        self.params.load_state_dict(state_dict)

    def save_net(self, new_save=False):
        T.save(self.params.state_dict(), file_path(dir=r"models\ppo_models", new_save=new_save))

    def load_net(self, path=None):
        try:
            self.params.load_state_dict(T.load(file_path(dir=r"models\ppo_models", new_save=False)))
            if path is not None:
                self.params.load_state_dict(T.load(path))
        except FileNotFoundError:
            print("Error while loading model.")
            return


@ray.remote
def train_play(n_iterations, ps: ParameterServer):
    scores, apples, wins = [], [], []
    mem = Memory()
    agent = Agent(lr_actor=1e-4, lr_critic=2e-4, gamma=0.99, K_epochs=10, eps_clip=0.1, gpu=True)
    agent.policy_old.load_state_dict(ray.get(ps.get_params.remote()))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=False)
    for i in range(1, n_iterations + 1):
        score = 0
        won = False
        around_view, cat_obs = game.reset()
        while not game.game.p.done:
            around_view, cat_obs, action, log_probs = agent.policy_old.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)

            mem.add(around_view, cat_obs, action, log_probs, reward, done)

            score += reward
            around_view = around_view_new
            cat_obs = cat_obs_new

        scores.append(score)
        apples.append(game.game.p.apple_count)
        wins.append(won)

    T.cuda.empty_cache()
    return mem, mean(scores), mean(apples), sum(wins)


def train(mem: Memory, ps: ParameterServer, agent: Agent):
    agent.policy.load_state_dict(ray.get(ps.get_params.remote()))
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.learn(mem)
    ps.update_params.remote(agent.policy.state_dict())
    mem.clear_memory()
    T.cuda.empty_cache()


if __name__ == '__main__':
    start_time = time.time()
    scores, apple_scores, wins = [], [], []
    played_games, n_iterations = 10, 10
    ray.init(num_cpus=3)
    agent = Agent(lr_actor=1e-3, lr_critic=1.5e-3, gamma=0.99, K_epochs=10, eps_clip=0.10, gpu=True)
    actor_scheduler = ExponentialLR(agent.policy.base_actor.optimizer, 0.8, verbose=True)
    critic_scheduler = ExponentialLR(agent.policy.base_critic.optimizer, 0.8, verbose=True)
    ps = ParameterServer.remote()
    played_games_id = ray.put(played_games)
    iter_time = time.time()
    futures = [train_play.remote(played_games_id, ps) for _ in range(n_iterations)]
    j = 0
    while len(futures):
        done_id, futures = ray.wait(futures)
        mem, mean_score, mean_apple, sum_wins = ray.get(done_id[0])
        scores.append(mean_score)
        apple_scores.append(mean_apple)
        wins.append(sum_wins)
        train(mem, ps, agent)

        if j >= 50 and j % 50 == 0:
            m, b, _, _, _ = linregress(list(range(50)), apple_scores[-50:])
            if m <= 0:
                actor_scheduler.step()
                critic_scheduler.step()
                print("\n")

        if j % 100 == 0:
            ps.save_net.remote(j == 0)
        j += 1
        t = time.time()

        time_step = str(datetime.timedelta(seconds=int(t - iter_time) * (n_iterations - j)))
        passed_time = str(datetime.timedelta(seconds=int(t - start_time)))
        iter_time = time.time()
        suffix_1 = f"Passed Time: {passed_time} | Remaining Time: {time_step}"
        suffix_2 = f" | Food_avg: {round(mean(apple_scores[-5:]), 2)} | Score_avg: {round(mean(scores[-5:]), 2)}"
        suffix_3 = f" | Wins:{int(mean(wins[-4:]))}"
        print_progress(j, n_iterations, suffix=suffix_1 + suffix_2 + suffix_3)

    print(f"Train Time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
    ps.save_net.remote(False)
    save_worked = False
    while not save_worked:
        try:
            agent.policy.load_state_dict(ray.get(ps.get_params.remote()))
            save_worked = True
        except Exception:
            ps.save_net.remote(False)
    ray.shutdown()
    path_ = file_path(dir=r"stats\ppo_stats", new_save=True, file_name="stats")
    plot_learning_curve([i + 1 for i in range(len(scores))], apple_scores, scores, path_)
