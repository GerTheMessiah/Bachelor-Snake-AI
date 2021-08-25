import gym
from src.snakeAI.gym_game.snake_env.snake_game import SnakeGame
from src.common.stop_game_exception import StopGameException


class SnakeEnv(gym.Env):
    def __init__(self, shape=(8, 8), has_gui=False):
        self.shape = shape
        self.has_gui = has_gui
        self.game = SnakeGame(self.shape, self.has_gui)

    def step(self, action, reward_function="standard"):
        self.game.action(action=action)
        around_view, scalar_obs = self.game.observe()
        reward = self.game.evaluate(reward_function=reward_function)
        done = self.game.is_done
        return around_view, scalar_obs, reward, done, self.game.max_snake_length == self.game.p.apple_count + 1  # has won

    def reset(self):
        self.game.reset_snake_game()
        around_view, scalar_obs = self.game.observe()
        return around_view, scalar_obs

    def render(self, close=False):
        self.game.view()

    def close(self):
        raise StopGameException()

    @property
    def has_ended(self):
        return self.game.p.done

    @property
    def apple_count(self):
        return self.game.p.apple_count
