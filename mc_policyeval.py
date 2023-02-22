import random, matplotlib
import matplotlib.pyplot as plt
from gridworld import *


def equiprobable_policy():
    choice = random.choice([0,1,2,3])
    return choice


def mc_evaluate(world, iters=100):

    world_visits = np.zeros([world.rows, world.columns])
    world_cumulative_sum = np.zeros([world.rows, world.columns])

    for i in range(iters):
        print('iteration:  ', i)
        for h in range(world.rows):
            for w in range(world.columns):
                if (h, w) in world.walls:
                    continue
                # reinitialize and start in new position
                world.is_terminal = False
                world.state = (h, w)
                states = []

                while not world.is_terminal:    # for each iteration, we play a game until terminal state is reached
                    world.state = world.next_state(equiprobable_policy())
                    reward = world.get_reward()
                    world.terminal()
                    states.append((reward, world.state, world.is_terminal))

                G = 0
                for reward, state, _ in states[::-1]:
                    G += reward
                    world_visits[state[0], state[1]] += 1
                    world_cumulative_sum[state[0], state[1]] += G

    return world_cumulative_sum / world_visits
