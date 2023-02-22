import random, math


class Node:
    def __init__(self, level, name, value=None):
        self.level = level
        self.value = value
        self.left = None
        self.right = None
        self.name = name

        # stuff for calculating MCTS
        self.visits = 0
        self.cumulative_sum = 0
        self.UCB = 0


class Tree:
    def __init__(self, depth):
        self.depth = depth
        self.target_node = self.random_target()
        self.root = Node(level=0, name='', value=random.random())
        self.populate(self.root, self.root.name)

    def populate(self, node, name):
        if node.level < self.depth:
            if not node.left:
                new_name = name + 'L'
                node.left = Node(level=node.level+1, name=new_name, value=random.random())
                self.populate(node.left, new_name)
            if not node.right:
                new_name = name + 'R'
                node.right = Node(level=node.level+1, name=new_name, value=random.random())
                self.populate(node.right, new_name)
        else:
            d = edit_distance(node.name, self.target_node)
            B = 1
            ro = 5
            node.value = B * math.e ** (-d / ro)
        return

    def random_target(self):
        name = ''
        for i in range(self.depth):
            if random.random() > 0.5:
                name = name + 'L'
            else:
                name = name + 'R'
        return name

def edit_distance(inst, target):
    if len(target) != len(inst):
        print(':::ERROR::: Strings have different lengths!')
        print(inst)
        print(target)
        return
    dist = 0
    for i in range(len(target)):
        if target[i] != inst[i]:
            dist += 1
    return dist


def rollout(node):
    # this is a specific case where we know that at each level there will be two further options,
    # otherwise rollout is blind and would just select randomly from any given children nodes
    while node.left and node.right:
        if random.random() > 0.5:
            node = node.left
        else:
            node = node.right
    return node.value
