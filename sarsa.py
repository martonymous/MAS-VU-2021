from gridworld import *


class SARSA_QLearning:
    def __init__(self, learning_rate=0.1, exploration_rate=0.1, gamma=0.999, q_learn=False):
        self.actions = [0, 1, 2, 3]
        self.State = Gridworld()
        self.lr = learning_rate
        self.epsilon = exploration_rate
        self.gamma = gamma
        self.q_vals = np.zeros((self.State.rows, self.State.columns, len(self.actions)))
        for wall in self.State.walls:
            self.q_vals[wall[0], wall[1]] = math.nan

        # this only signifies the difference between SARSA and Q-learning, i.e. Q is updated differently
        self.Q = q_learn

    def choose_action(self, new_state=None):
        # exploration
        if random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(self.actions)

        # otherwise choose greedy action
        # sometimes, multiple actions can have same reward, in that case choose randomly from those actions
        else:
            if not new_state:
                state = self.State.state
            else:
                state = new_state

            max_reward = max(self.q_vals[state[0], state[1], :])
            best_actions = [a for a in self.actions if self.q_vals[state[0], state[1], a] == max_reward]
            action = random.choice(best_actions)
        return action

    def take_action(self, action):
        return self.State.next_state(action)

    def update_q(self, prev_state, prev_action, reward, state, action=None):
        # update Q in SARSA or Q-Learning, formula taken from Sutton and Barto
        if not self.Q:
            reward_plus = reward + (self.gamma * self.q_vals[state[0], state[1], action])   # SARSA
        else:
            reward_plus = reward + (self.gamma * self.q_vals[state[0], state[1], :].max())  # Q-Learning

        self.q_vals[prev_state[0], prev_state[1], prev_action] += self.lr * (reward_plus - self.q_vals[prev_state[0], prev_state[1], prev_action])

    def reset(self):
        self.State = Gridworld()

    def run_s(self, iterations=100):
        for i in range(iterations):
            print(i)
            action = self.choose_action()
            while not self.State.is_terminal:
                # take action and get new state, next action (for SARSA only) and reward to update Q
                next_state = self.take_action(action)
                next_action = self.choose_action(next_state)
                reward = self.State.get_reward(next_state)

                self.update_q(prev_state=self.State.state, prev_action=action, reward=reward,
                              state=next_state, action=next_action)

                # new actions become previous actions for next step
                action = next_action
                self.State.state = next_state

                # check if terminal state is reached
                self.State.terminal()

            self.reset()

    def run_q(self, iterations=100):
        for i in range(iterations):
            print(i)
            while not self.State.is_terminal:
                # take action and get new state, next action (for SARSA only) and reward to update Q
                action = self.choose_action()
                next_state = self.take_action(action)
                reward = self.State.get_reward(next_state)

                self.update_q(prev_state=self.State.state, prev_action=action, reward=reward,
                              state=next_state)
                self.State.state = next_state

                # check if terminal state is reached
                self.State.terminal()

            self.reset()
