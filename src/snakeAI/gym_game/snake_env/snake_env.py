import gym
from src.snakeAI.gym_game.snake_env.snake_game import SnakeGame
from src.common.stop_game_exception import StopGameException


class SnakeEnv(gym.Env):
    def __init__(self, shape=(8, 8), has_gui=False):
        self.shape = shape
        self.has_gui = has_gui
        self.game = SnakeGame(self.shape, self.has_gui)

    """
    This method is processing a step in the environment.
    @:param action: Action (0 -> turn left, 1 -> turn right, 2 -> do nothing)
    @:param reward_function: Reward function (standard function, optimized function).
    """
    def step(self, action, reward_function="standard"):
        self.game.action(action=action)
        around_view, scalar_obs = self.game.observe()
        reward = self.game.evaluate(reward_function=reward_function)
        is_terminal = self.game.is_terminal
        return around_view, scalar_obs, reward, is_terminal, self.game.max_snake_length == self.game.p.apple_count + 1  # has won

    """
    This method resets the environment.
    @:param new_shape: New Shape of the playground.
    """
    def reset(self, new_shape=None):
        self.game.reset_snake_game(new_shape=new_shape)
        around_view, scalar_obs = self.game.observe()
        return around_view, scalar_obs

    """
    This method is rendering the environment.
    @:param close: Useless.
    """
    def render(self, close=False):
        self.game.view()

    """
    This method is closing the environment together with the gui.
    """
    def close(self):
        raise StopGameException()
    """
    @:return A boolean whether the has_ended.
    """
    @property
    def has_ended(self):
        return self.game.p.is_terminal

    """
    @:return Number of the collected apple in a game.
    """
    @property
    def apple_count(self):
        return self.game.p.apple_count
