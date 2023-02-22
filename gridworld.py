import random, math
import numpy as np
import itertools


class Gridworld:
    def __init__(self, rows=9, columns=9, win_position=(8, 8), lose_position=(6, 5)):
        self.rows = rows
        self.columns = columns
        self.win_position = win_position
        self.lose_position = lose_position
        self.walls = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                              (2, 6), (3, 6), (4, 6), (5, 6),
                              (7, 1), (7, 2), (7, 3), (7, 4)]
        self.board = np.zeros([rows, columns])
        for wall in self.walls:
            self.board[wall[0], wall[1]] = math.nan

        starting_positions = list(itertools.product(range(rows), range(columns)))
        self.starting_positions = [pos for pos in starting_positions if (pos not in self.walls)]
        self.starting_positions.remove(self.win_position)
        self.starting_positions.remove(self.lose_position)
        self.state = random.choice(self.starting_positions)
        self.is_terminal = False

    def get_reward(self, new_state=None):
        if new_state == None:
            state = self.state
        else:
            state = new_state

        if state == self.win_position:      # terminal state reward
            return 50
        elif state == self.lose_position:   # terminal state reward
            return -50
        else:
            return -2                       # step penalty

    def terminal(self):
        if (self.state == self.win_position) or (self.state == self.lose_position):
            self.is_terminal = True

    def next_state(self, action):
        # up
        if action == 0:
            nextState = (self.state[0] - 1, self.state[1])
        # down
        elif action == 1:
            nextState = (self.state[0] + 1, self.state[1])
        # left
        elif action == 2:
            nextState = (self.state[0], self.state[1] - 1)
        # right
        else:
            nextState = (self.state[0], self.state[1] + 1)

        # if the next state is permitted, i.e. no wall and not out-of-bounds, then proceed to next state
        if (nextState[0] >= 0) and (nextState[0] <= (self.rows-1)):
            if (nextState[1] >= 0) and (nextState[1] <= (self.columns-1)):
                if nextState not in self.walls:
                    return nextState
        return self.state
