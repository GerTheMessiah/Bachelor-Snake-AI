import numpy as np
from dataclasses import dataclass
from random import randint
from src.snakeAI.gym_game.snake_env.gui import GUI
from src.snakeAI.gym_game.snake_env.observation import make_obs
from src.snakeAI.gym_game.snake_env.reward import Reward


@dataclass()
class Player:
    pos: np.ndarray
    tail: list
    direction: int
    id: int
    c_s: int
    c_h: int
    inter_apple_steps: int
    is_terminal: bool

    def player_reset(self, pos: np.ndarray):
        self.pos = pos
        self.tail.clear()
        self.tail.append((pos[0], pos[1]))
        self.direction = randint(0, 3)
        self.inter_apple_steps = 0
        self.is_terminal = False

    @property
    def apple_count(self):
        return len(self.tail) - 1

    @property
    def last_tail_pos(self):
        return np.array(self.tail[-1]) if not np.equal(np.array(self.tail[-1]), self.pos).all() else None

    @property
    def snake_len(self):
        return len(self.tail)


class SnakeGame:
    def __init__(self, shape, has_gui):
        self.shape = shape
        self.ground = np.zeros((self.shape[0], self.shape[1]), dtype=np.int8)
        pos = np.array((randint(0, self.shape[0] - 1), randint(0, self.shape[1] - 1)))
        self.p = Player(pos=pos, tail=[(pos[0], pos[1])], direction=randint(0, 3), id=1, c_s=1, c_h=2,
                        inter_apple_steps=0, is_terminal=False)
        self.reward = Reward(self)

        self.has_gui = has_gui
        self.step_counter = 0
        self.ground[pos[0], pos[1]] = self.p.c_h
        self.apple = self.make_apple()
        if has_gui:
            self.gui = GUI(self.shape)

    def action(self, action):
        if self.p.inter_apple_steps >= self.max_snake_length:
            self.p.is_terminal = True
            return

        self.p.inter_apple_steps += 1
        self.step_counter += 1
        if action == 0:
            self.p.direction = (self.p.direction + 1) % 4

        elif action == 1:
            self.p.direction = (self.p.direction - 1) % 4

        else:
            pass

        ########################## step ##########################
        self.p.pos[self.p.direction % 2] += -1 if self.p.direction % 3 == 0 else 1
        if not all(0 <= self.p.pos[i] < self.ground.shape[i] for i in range(2)):
            self.p.is_terminal = True
            return
        self.p.tail.insert(0, (self.p.pos[0], self.p.pos[1]))


        # if has won return method
        if len(self.p.tail) == self.max_snake_length:
            self.p.is_terminal = True
            return

        # if snake has eaten remove apple
        if self.p.tail[0] == self.apple:
            self.ground[self.p.tail[0][0], self.p.tail[0][1]] = self.p.c_h
            self.apple = self.make_apple()
            self.p.inter_apple_steps = 0
            self.reward.has_grown = True
        else:
            self.ground[self.p.tail[-1][0], self.p.tail[-1][1]] = 0
            del self.p.tail[-1]
            self.reward.has_grown = False
        # prof lost
        if len(self.p.tail) != len(set(self.p.tail)):
            self.p.is_terminal = True
            return

        if not self.p.is_terminal:
            for s in self.p.tail:
                self.ground[s[0], s[1]] = self.p.c_s
            self.ground[self.p.tail[-1][0], self.p.tail[-1][1]] = -1
            self.ground[self.p.tail[0][0], self.p.tail[0][1]] = self.p.c_h

    def evaluate(self, reward_function=None):
        if reward_function is "A":
            return self.reward.optimized_reward
        else:
            return self.reward.standard_reward

    def view(self):
        if self.has_gui:
            self.gui.update_GUI(self.ground)

    def observe(self):
        return make_obs(self.p.id, self.p.pos, self.p.last_tail_pos, self.p.direction, self.ground, self.apple,
                        self.p.inter_apple_steps)

    def make_apple(self):
        pos = np.where(self.ground == 0)
        if pos[0].size != 0:
            rand = randint(0, len(pos[0]) - 1)
            apple = (pos[0][rand], pos[1][rand])
            self.ground[apple[0], apple[1]] = -2
        else:
            return None
        return apple

    def reset_snake_game(self, new_shape=None):
        if new_shape is not None:
            self.shape = new_shape
            del self.ground
            self.ground = np.zeros((self.shape[0], self.shape[1]), dtype=np.int8)
        else:
            self.ground.fill(0)
        pos = np.array((randint(0, self.shape[0] - 1), randint(0, self.shape[1] - 1)))
        self.p.player_reset(pos)
        self.step_counter = 0
        self.reward.has_grown = False
        self.ground[pos[0], pos[1]] = self.p.c_h
        self.apple = self.make_apple()
        if self.has_gui:
            self.gui.reset_GUI()

    @property
    def max_snake_length(self):
        return self.ground.size

    @property
    def is_terminal(self):
        return self.p.is_terminal
