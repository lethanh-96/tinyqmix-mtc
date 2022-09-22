import numpy as np
import torch

class Controller:

    def __init__(self, env, args):
        # save data
        self.args  = args
        self.env   = env
        self.state = np.random.randint(low=0,
                                        high=args.n_rb,
                                        size=args.n_node)
        # initialize q table
        self.q_table = np.ones([args.n_node, args.n_rb, args.n_rb])
        # initialize intermitten learning params
        self.count  = 0
        self.reward = []
        self.action = self.state
        self.eps    = self.args.eps_end

    def select_np_action(self, _):
        return self.action

    def update(self, action, reward):
        if self.count < 100:
            self.count  += 1
            self.reward.append(reward)
        else:
            self.reward.append(reward)
            self.reward = np.mean(self.reward)
            # extract data
            state         = self.state
            next_state    = action
            args          = self.args
            gamma         = 0.1
            learning_rate = 0.1
            eps           = 0.01

            # update q table
            for i in range(self.args.n_node):
                self.q_table[i, state[i], action[i]] += \
                    learning_rate * (\
                        self.reward + gamma * np.max(self.q_table[i, next_state[i]])\
                        - self.q_table[i, state[i], action[i]]
                    )

            # update state
            self.state = next_state

            # reset intermittent learning params
            self.count = 0
            self.reward = []
            state  = self.state
            action = np.zeros(self.args.n_node)
            for i in range(self.args.n_node):
                if np.random.rand() < self.eps:
                    action[i] = np.random.randint(low=0, high=self.args.n_rb)
                else:
                    action[i] = np.argmax(self.q_table[i, state[i]])
            self.action = action.astype(np.int32)
            
            self.eps *= 0.99
