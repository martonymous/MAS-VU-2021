import matplotlib.pyplot as plt
import numpy as np
import matplotlib

"""
Plotting tools generously provided by fellow student Oleg. Slight adjustments from my side.
"""


def normalize(vec):
    vec = vec - vec.min()
    return vec / vec.sum()


def plot_qvals(env, q_table, title='', episode_num=100):
    fig = plt.figure(figsize=(14, 8))

    cmap = matplotlib.cm.rainbow

    v_table_optimal = np.max(q_table, axis=2)
    im = plt.imshow(v_table_optimal, cmap=cmap, interpolation='nearest', origin='upper')

    for x in range(q_table.shape[0]):  # inverted because of imshow
        for y in range(q_table.shape[1]):  # inverted because of imshow
            state_q_values = q_table[x, y]
            #             print(f"x={x}, y={y}, state_q_values={state_q_values}")
            state_q_values_norm = normalize(state_q_values)
            #             print(f"x={x}, y={y}, state_q_values_norm={state_q_values_norm}")

            optimal_q = state_q_values_norm.max()
            for i, q in enumerate(state_q_values_norm):
                if q < optimal_q:  # if q < 0.3:
                    continue
                if i == 0:  # UP
                    plt.arrow(y, x, 0, -0.5 * q, fill=False,
                              length_includes_head=True, head_width=0.1,
                              alpha=0.8, color='k')
                if i == 2:  # LEFT
                    plt.arrow(y, x, -0.5 * q, 0, fill=False,
                              length_includes_head=True, head_width=0.1,
                              alpha=0.8, color='k')
                if i == 1:  # DOWN
                    plt.arrow(y, x, 0, 0.5 * q, fill=False,
                              length_includes_head=True, head_width=0.1,
                              alpha=0.8, color='k')
                if i == 3:  # RIGHT
                    plt.arrow(y, x, 0.5 * q, 0, fill=False,
                              length_includes_head=True, head_width=0.1,
                              alpha=0.8, color='k')

    plt.title((("" or title) + "\n") + f"Optimal values and state actions; Num Episodes={episode_num}, "
                                       f"LR={env.lr}, Epsilon={env.epsilon}, GAMMA={env.gamma}")
    plt.grid(b=False)
    plt.colorbar(im, orientation='vertical')
    plt.show()