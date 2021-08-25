import numpy as np


class Reward:
    def __init__(self, snake_game):
        self.has_grown = False
        self.snakeGame = snake_game

    @property
    def standard_reward(self):
        if len(self.snakeGame.p.tail) == self.snakeGame.max_snake_length and self.snakeGame.p.done:
            return 100
        elif len(self.snakeGame.p.tail) != self.snakeGame.max_snake_length and self.snakeGame.p.done:
            return -10
        elif self.has_grown:
            return 2.5
        else:
            return -0.01

    @property
    def optimized_reward(self):
        len = self.snakeGame.max_snake_length
        r_distance = 10 / np.linalg.norm(self.snakeGame.p.pos - np.array(self.snakeGame.apple)) * (self.snakeGame.p.snake_len / len)

        r_timeout = (1 / len) * (self.snakeGame.p.inter_apple_steps / self.snakeGame.p.snake_len)
        return r_distance - r_timeout
