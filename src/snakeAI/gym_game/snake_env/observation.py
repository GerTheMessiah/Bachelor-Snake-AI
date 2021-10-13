import numpy as np

"""
This function proofs whether a and b are on playground.
@:param a: Coordinate of the x-axis.
@:param b: Coordinate of the y-axis.
@:param size: Playground size.
"""
def on_playground(a, b, size):
    return (0 <= a < size[0]) and (0 <= b < size[1])

"""
This function determines a inverted distance.
@:param ground: playground.
@:param p_pos: player position.
@:param wanted: List of elements which should be found.
@:param fac_0: Factor for controlling the north / south direction.
@:param fac_1: Factor for controlling the east / west direction.
"""
def dist(ground, p_pos, wanted, fac_0, fac_1):
    dist_, i_0, i_1 = 0, 1, 1
    p_0 = p_pos[0] + fac_0 * i_0
    p_1 = p_pos[1] + fac_1 * i_1
    while on_playground(p_0, p_1, ground.shape) and ground[p_0, p_1] not in wanted:
        i_0 += 1
        i_1 += 1
        dist_ += 1
        p_0 = p_pos[0] + fac_0 * i_0
        p_1 = p_pos[1] + fac_1 * i_1
    if not on_playground(p_0, p_1, ground.shape) and bool(wanted):
        return 0
    return 1 / dist_ if dist_ != 0 else 2

"""
This function is creating the around_view.
@:param pos: Position of the snake head.
@:param id: Id. Important for the detection of the snake head.
@:param g: ground -> Playground.
"""
# 1x6x13x13
def create_around_view(pos, id, g):
    width = 6
    c_h = id * 2
    c_s = id
    tmp_arr = np.zeros((6, width * 2 + 1, width * 2 + 1), dtype=np.float64)

    for row in range(-width, width + 1):
        for column in range(-width, width + 1):
            if on_playground(pos[0] + row, pos[1] + column, g.shape):
                a, b = pos[0] + row, pos[1] + column
                if g[a, b] == c_s:
                    tmp_arr[1, row + width, column + width] = 1
                    continue

                elif g[a, b] == c_h:
                    tmp_arr[2, row + width, column + width] = 1
                    continue

                elif g[a, b] == 0:
                    tmp_arr[3, row + width, column + width] = 1
                    continue

                elif g[a, b] == -1:  # End of snake tail.
                    tmp_arr[4, row + width, column + width] = 1
                    continue

                elif g[a, b] == -2:
                    tmp_arr[5, row + width, column + width] = 1

            else:
                tmp_arr[0, row + width, column + width] = 1

    return np.expand_dims(tmp_arr, axis=0)

"""
This method is creating the distance observation.
@:param pos: Position of the snake head.
@:param ground: Playground.
"""
# a = 24
def create_distances(pos, ground):
    obs = np.zeros(24, dtype=np.float64)
    a = 0
    grad_list = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    for wanted in [[], [-1, 1, 2], [-2]]:
        for grad in grad_list:
            obs[a] = dist(ground, pos, wanted, *grad)
            a += 1
    return obs

"""
The method is creating the direction observation.
@:param direction: Direction of the snake.
"""
# a + 4
def direction_obs(direction):
    obs = np.zeros(4, dtype=np.float64)
    obs[0 + direction] = 1
    return obs

"""
This method creates the two compass observations (apple, snake head) and (last element of the snake tail, snake head).
@:param pos: Position of the snake head.
@:param obj: Object. Snake head or last element of the snake tail.
"""
# a + 6
def compass_obs(pos, obj):
    obs = np.zeros(6, dtype=np.float64)
    if obj is None:
        return obs
    obs[0] = 1 if pos[0] < obj[0] else 0
    obs[1] = 1 if pos[1] > obj[1] else 0
    obs[2] = 1 if pos[0] > obj[0] else 0
    obs[3] = 1 if pos[1] < obj[1] else 0
    obs[4] = 1 if pos[0] == obj[0] else 0
    obs[5] = 1 if pos[1] == obj[1] else 0
    return obs

"""
This method creates the hunger observation.
@:param inter_apple_steps: Steps since an apple was eaten.
@:param size: Size of the playground.
"""
# a + 1
def hunger_obs(inter_apple_steps, size):
    obs = np.zeros(1, dtype=np.float64)
    obs[0] = 1 / (size - inter_apple_steps) if inter_apple_steps != size else 2
    return obs

"""
This method is generating the whole observation.
@:param p_id: Player id.
@:param pos: Position of the snake head.
@:param tail_pos: Position of the last snake tail element.
@:param direction: Direction of the snake head.
@:param ground: Playground.
@:param food: Apple position.
@:param iter_apple_counter: Steps since an apple was eaten.
"""
def make_obs(p_id, pos, tail_pos, direction, ground, food, iter_apple_counter):
    around_view = create_around_view(pos, p_id, ground)
    distances = create_distances(pos, ground)
    direction = direction_obs(direction)
    apple_obs = compass_obs(pos, food)
    tail_obs = compass_obs(pos, tail_pos)
    hunger = hunger_obs(iter_apple_counter, ground.size)
    scalar_obs = np.concatenate((distances, direction, apple_obs, hunger, tail_obs))
    return around_view, np.expand_dims(scalar_obs, axis=0)
