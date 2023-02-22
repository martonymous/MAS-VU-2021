# from tree import *
# from mcts import *
from gridworld import *
from mc_policyeval import *
from sarsa import *
from plotting import *


""" Part 3 - MC evaluation """
q1 = mc_evaluate(world=Gridworld())
cmap = matplotlib.cm.rainbow
im = plt.imshow(q1, cmap=cmap, origin='upper')
plt.colorbar(im)
plt.show()

""" SARSA """
sarsa = SARSA_QLearning(q_learn=False)
sarsa.run_s(500)
plot_qvals(sarsa, sarsa.q_vals, title='Gridworld example with SARSA', episode_num=500)

""" Q-Learning"""
sarsa = SARSA_QLearning(q_learn=True)
sarsa.run_s(500)
plot_qvals(sarsa, sarsa.q_vals, title='Gridworld example with Q-Learning', episode_num=500)
