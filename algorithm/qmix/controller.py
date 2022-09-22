from tqdm import tqdm

import numpy as np
import pickle
import torch
import math
import os

from .replay_memory import ReplayMemory
from .preprocessor import Preprocessor
from .transition import Transition
from .mixer import Mixer
from .agent import Agent

class Controller:

    def __init__(self, env, args):
        self.args = args
        self.env  = env
        self.global_step = 0
        self.episode = 0
        self.best_avg_reward = - np.inf

        self.agents = [Agent(env, args) for _ in range(args.n_node)]
        self.mixer  = Mixer(self.agents, args).to(args.device)

        self.__init_preprocessor()
        self.__init_optimizer()
        self.__init_replay_memory()

    def __init_preprocessor(self):
        self.preprocessor = Preprocessor(self.args).to(self.args.device)

    def __init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.mixer.parameters(), 
                                          lr=self.args.learning_rate)

    def __init_replay_memory(self):
        self.replay_memory = ReplayMemory(self.args.memory_size)

    @property
    def eps_threshold(self):
        if self.args.mode == 'train':
            return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1. * self.global_step / self.args.eps_decay)
        else:
            return self.args.eps_end

    def add_action_noise(self, action, n_action):
        ra = torch.randint(low=0, high=n_action, size=(self.args.n_node,), device=self.args.device)
        r  = torch.rand(self.args.n_node)
        idx = r < self.eps_threshold
        action[idx] = ra[idx]
        return action

    def add_action_noise_v2(self, action, qs):
        second_best_action = torch.topk(qs, 2, dim=1)[1][:, 1]
        r = torch.rand(self.args.n_node)
        idx = r < self.eps_threshold
        action[idx] = second_best_action[idx]
        return action

    def select_action(self, states):
        if self.args.mode == 'train':
            states = self.preprocessor.fit_transform(states)
        else:
            states = self.preprocessor.transform(states)
        qs = self.mixer.compute_policy_qs(states)
        actions = qs.max(1)[1]
        # actions = self.add_action_noise(actions, self.args.n_rb)
        actions = self.add_action_noise_v2(actions, qs)
        return actions

    def select_np_action(self, states):
        return self.select_action(states).detach().cpu().numpy()

    def run(self, monitor):
        # initialize
        states  = self.env.reset()
        losses  = []
        Rewards = []

        for step in range(self.args.n_step * self.args.n_train_episode):
            # select action
            actions = self.select_action(states)

            # convert action to cpu
            np_actions = actions.detach().cpu().numpy()

            # step
            next_states, rewards, done, info = self.env.step(np_actions)

            # store the transition in memory
            self.replay_memory.push(states, actions, next_states, rewards)

            # move to next observation
            states = next_states

            # perform one step of the optimization on policy network
            if step % self.args.train_frequency == 0:
                losses = []
                for _ in range(self.args.train_frequency):
                    loss = self.optimize_policy()
                    losses.append(loss)
                self.update_target_net()
            self.global_step += 1

            if done:
                break

            # update monitor
            info['eps'] = self.eps_threshold
            if len(losses) > 0:
                if losses[0] is not None:
                    info['loss'] = np.mean(losses)
                else:
                    info['loss'] = 0
            else:
                info['loss'] = 0
            info['memory_size'] = len(self.replay_memory)
            monitor.step(info)

            # update arrival model after an episode end
            if step % self.args.n_step == 0:
                states = self.env.reset()

            # record rewards
            Rewards.append(float(torch.mean(rewards).item()))
            if step % (self.args.n_step * self.args.save_model_iter) == 0:
                self.save(monitor)
                avg_reward = np.mean(Rewards)
                Rewards = []
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    self.save_best(monitor)
                    print(f'[+] saved new best model: {avg_reward}')

    def optimize_policy(self):
        # check if enough data or not
        if len(self.replay_memory) < self.args.batch_size:
            return

        # sample transitions from replay memory
        transitions = self.replay_memory.sample(self.args.batch_size)

        # transpose the batch
        batch = Transition(*zip(*transitions))

        # prepare batch data
        next_states = torch.stack(batch.next_state)
        next_states = self.preprocessor.transform(next_states)
        states      = torch.stack(batch.state)
        states      = self.preprocessor.transform(states)
        actions     = torch.stack(batch.action).unsqueeze(2)
        rewards     = torch.stack(batch.reward)

        # compute Q(s_t, a) the model computes Q(s_t), then we select the
        # columns of actions taken.
        qs    = self.mixer.compute_policy_qs(states)
        qs    = qs.gather(2, actions).squeeze()
        q_tot = self.mixer.compute_q_tot(qs, states)

        # compute V(s_{t+1}) for all next states
        next_qs    = self.mixer.compute_target_qs(next_states)
        next_qs    = next_qs.max(2)[0].detach()
        next_q_tot = self.mixer.compute_q_tot(next_qs, next_states)

        # compute expected Q value
        expected_q_tot = (next_q_tot * self.args.gamma) + rewards

        # compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(q_tot, expected_q_tot)
        loss_item = float(loss.item())

        # optimizer the model
        self.optimizer.zero_grad()
        loss.backward()
        # for agent in self.agents[:1]:
        for agent in self.agents:
            for param in agent.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss_item

    def update_target_net(self):
        for agent in self.agents:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    def create_save_path(self, monitor):
        folder = os.path.join(self.args.model_dir, monitor.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, 'controller.pkl')
        return path

    def save(self, monitor):
        path = self.create_save_path(monitor)
        with open(path, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    def create_save_best_path(self, monitor):
        folder = os.path.join(self.args.model_dir, monitor.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, 'best_controller.pkl')
        return path

    def save_best(self, monitor):
        path = self.create_save_best_path(monitor)
        with open(path, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    def load(self, env, monitor):
        path = self.create_save_path(monitor)
        if not os.path.exists(path):
            controller = self
        else:
            with open(path, 'rb') as fp:
                controller = pickle.load(fp)
            print(f'[+] loaded from {path}, avg_reward={controller.best_avg_reward}')
        controller.env  = env
        controller.args = env.args
        controller.optimizer = torch.optim.Adam(controller.mixer.parameters(), 
                                                lr=env.args.learning_rate)

        if self.args.mode == 'train':
            controller.episode += 1
        return controller

    def load_best(self, env, monitor):
        path = self.create_save_best_path(monitor)
        # print(f'[+] try to load best model from {path}')
        if not os.path.exists(path):
            controller = self
            # print(f'    - not exists')
        else:
            with open(path, 'rb') as fp:
                controller = pickle.load(fp)
            print(f'    - loaded from {path}, avg_reward={controller.best_avg_reward}')
        controller.env  = env
        controller.args = env.args
        controller.optimizer = torch.optim.Adam(controller.mixer.parameters(), 
                                                lr=env.args.learning_rate)

        if self.args.mode == 'train':
            controller.episode += 1
        return controller
