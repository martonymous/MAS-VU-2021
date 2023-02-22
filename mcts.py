import itertools
from tree import *
import matplotlib.pyplot as plt

depth = 15
MC_tree = Tree(depth)

iterations = depth

c = [1,2,5]
root_iterations = [10,50]
rollout_iterations = [1,5]

num_runs = 30
experiments = list(itertools.product(c, root_iterations, rollout_iterations))
all_experiment_data = {}

for experiment in experiments:
    print(experiment)

    # experiment parameters
    c = experiment[0]
    root_iteration = experiment[1]
    rollout_iteration = experiment[2]

    exp_data = []

    for exp_run in range(num_runs):
        random.seed(exp_run)

        root = MC_tree.root
        decision_path = []
        for h in range(iterations):
            for i in range(root_iteration):
                next_node = root
                path = []
                # root: select next node
                # if all children visited, based on UCB (which is calculated on the spot)
                while next_node.left and next_node.right and next_node.left.visits > 0 and next_node.right.visits > 0:

                    """calculate UCBs"""
                    next_node.left.UCB = (next_node.left.cumulative_sum / next_node.left.visits) + (c * math.sqrt(math.log(next_node.visits) / next_node.left.visits))
                    next_node.right.UCB = (next_node.right.cumulative_sum / next_node.right.visits) + (c * math.sqrt(math.log(next_node.visits) / next_node.right.visits))

                    if next_node.left.UCB == next_node.right.UCB:
                        choice = random.choice(['left', 'right'])
                        if choice == 'left':
                            next_node = next_node.left
                        else:
                            next_node = next_node.right

                        path.append(choice)

                    elif next_node.left.UCB > next_node.right.UCB:
                        next_node = next_node.left
                        path.append('left')

                    else:
                        next_node = next_node.right
                        path.append('right')

                for i in range(rollout_iteration):
                    # if at least one child not visited, rollout from (random) unexplored child node
                    if not next_node.left and not next_node.right:
                        value = next_node.value

                    elif next_node.left.visits == 0 and next_node.right.visits == 0:
                        choice = random.choice(['left', 'right'])
                        if choice == 'left':
                            rand_node = next_node.left
                        else:
                            rand_node = next_node.right
                        path.append(choice)
                        value = rollout(rand_node)
                    elif next_node.left.visits == 0:
                        path.append('left')
                        value = rollout(next_node.left)
                    else:
                        path.append('right')
                        value = rollout(next_node.right)

                    # update path values (for UCB)
                    root.cumulative_sum += value
                    root.visits += 1
                    next_node = root
                    for split in path:
                        if split == 'left':
                            next_node = next_node.left
                        else:
                            next_node = next_node.right
                        next_node.cumulative_sum += value
                        next_node.visits += 1

            # after a number of iterations starting from same root, a new root may be selected based on UCB values
            root.left.UCB = (root.left.cumulative_sum / root.left.visits) + (c * math.sqrt(math.log(root.visits) / root.left.visits))
            root.right.UCB = (root.right.cumulative_sum / root.right.visits) + (c * math.sqrt(math.log(root.visits) / root.right.visits))

            if root.left.UCB == root.right.UCB:
                choice = random.choice(['L', 'R'])
                if choice == 'L':
                    root = root.left
                else:
                    root = root.right

                decision_path.append(choice)

            elif root.left.UCB > root.right.UCB:
                decision_path.append('L')
                root = root.left
            else:
                decision_path.append('R')
                root = root.right

        chosen_path = ''.join(decision_path)
        d = edit_distance(chosen_path, MC_tree.target_node)
        exp_data.append(d)

    all_experiment_data[experiment] = exp_data


data = [all_experiment_data[key] for key in all_experiment_data.keys()]

fig, ax = plt.subplots()
ax.set_title('Edit Distance of Chosen Leaf Node to Target Leaf Node for Different Experimental Configurations')
ax.boxplot(data, labels=[key for key in all_experiment_data.keys()])

plt.show()
