from gym import spaces
import numpy as np
import torch
import gym

from .state_lstm import StateMonitorLstm
from .reward import RewardMonitor
from .state import StateMonitor
from . import oma_rb
from . import ofdma

class Env(gym.Env):

    def __init__(self, args):
        super().__init__()
        # save data
        self.args = args
        # create network
        self.__create_network()

    # =====================================================================================
    # internal logic
    # =====================================================================================
    def __create_network(self):
        if self.args.protocol == 'oma_rb':
            self.net = oma_rb.Network(self.args)
        elif self.args.protocol == 'ofdma':
            self.net = ofdma.Network(self.args)
        else:
            raise NotImplementedError
        if self.args.algorithm == 'qmix_lstm':
            self.state_monitor = StateMonitorLstm(self.net)
        else:
            self.state_monitor = StateMonitor(self.net)
        self.reward_monitor = RewardMonitor(self.net, self.state_monitor)
        self.args.n_state   = self.state_monitor.get().shape[-1]
        print(f'[+] created network, n_state={self.args.n_state}')

    def initialize_info(self):
        self.info = {
            'traffic'      : np.zeros(self.args.n_node),
            'transmit'     : np.zeros(self.args.n_node),
            'success'      : np.zeros(self.args.n_node),
            'collision'    : np.zeros(self.args.n_node),
            'drop'         : np.zeros(self.args.n_node),
            'delay'        : np.zeros(self.args.n_node),
            'queue_length' : np.zeros(self.args.n_node),
            'cw'           : np.zeros(self.args.n_node),
            'traffic_mse'  : np.zeros(self.args.n_node),
            'success_mse'  : np.zeros(self.args.n_node),
        }

    def update_info(self, success, collision):
        self.info['traffic']      += self.net.n_traffic
        self.info['transmit']     += self.net.n_transmit
        self.info['success']      += success
        self.info['collision']    += collision
        self.info['drop']         += self.net.n_drop
        self.info['delay']        += self.net.peak_delay
        self.info['queue_length'] += self.net.queue_length
        self.info['cw']           += self.net.cw
        self.info['traffic_mse']  += self.state_monitor.traffic_estimation_mse
        self.info['success_mse']  += self.state_monitor.success_estimation_mse

    def __to_device(self, x):
        return torch.tensor(x, device=self.args.device, dtype=torch.float32)
    # =====================================================================================

    # =====================================================================================
    # API called by RL algorithm
    # =====================================================================================
    def reset(self):
        # reset network
        self.net.reset()
        state = self.state_monitor.get()
        state = self.__to_device(state)
        return state

    def step(self, action):
        # initialize
        if 'centralize' in self.args.protocol:
            pass
        else:
            self.initialize_info()
            self.reward_monitor.reset()
        # set cw if needed
        rbs = action
        self.net.set_rbs(rbs)
        # simulation over coherrent time
        if 'centralize' in self.args.protocol:
            iterator = range(6, self.args.coherrent_time)
        else:
            iterator = range(self.args.coherrent_time)
        for step in iterator:
            # generate traffic
            self.net.generate_traffic()
            # transmit packet
            self.net.transmit()
            # compute info
            success, collision = self.net.compute_info(rbs)
            # store info
            self.update_info(success, collision)
            # compute reward
            self.reward_monitor.update(step, rbs, success, collision)
            self.state_monitor.update(rbs, success, collision)
            # acknowledgement
            self.net.ack(collision)
            self.net.now += 1
        # compute reward
        reward = self.reward_monitor.get(rbs)
        self.info['reward'] = reward
        reward = self.__to_device(reward)
        # clean data
        self.net.clean()
        # compute next state
        state = self.state_monitor.get()
        state = self.__to_device(state)
        # compute info
        info = self.info
        info['state/traffic']   = 0 # state[0, 0]
        info['state/sr0']       = 0 # state[0, 2]
        info['state/sr1']       = 0 # state[0, 3]
        info['state/rbs']       = 0 # rbs[0]
        info['state/traffic_p'] = 0 # self.net.nodes[0].traffic_model.spatial_p(self.net.now) * self.args.lamda
        # compute done
        done = False
        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    # =====================================================================================

    # =====================================================================================
    # other properties
    # =====================================================================================
    @property
    def now(self):
        return self.net.now
    # =====================================================================================
