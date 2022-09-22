import numpy as np

from .node import Node
import traffic

class Network:

    def __init__(self, args):
        # save data
        self.args = args
        # create nodes        
        self.nodes = [Node(self) for idx in range(args.n_node)]
        # reset
        self.reset()        
        # time
        self.now = 0
        # 
        self.action = None

    def reset(self):
        # reset all nodes
        for node in self.nodes:
            node.reset()
        # set traffic models
        traffic.set_traffic_models(self, self.args)

    # =====================================================================================
    # internal logic
    # =====================================================================================
    def __compute_collision(self, action):
        # initialize
        collision = np.zeros(self.args.n_node)
        return collision

    # =====================================================================================
    # API called by environment
    # =====================================================================================
    def generate_traffic(self):
        for node in self.nodes:
            node.generate_traffic()

    def transmit(self):
        for node in self.nodes:
            node.transmit()

    def compute_info(self, actions):
        collision = self.__compute_collision(actions)
        success   = (1 - collision) * self.n_transmit
        return success, collision

    def ack(self, collision):
        for i, node in enumerate(self.nodes):
            node.ack(collision[i])

    def clean(self):
        for node in self.nodes:
            node.clean()

    def set_rbs(self, schedules):
        for i, schedule in enumerate(schedules):
            self.nodes[i].set_schedule(schedule)

    # =====================================================================================

    # =====================================================================================
    # other properties
    # =====================================================================================
    @property
    def n_transmit(self):
        return np.array([node.is_transmitting for node in self.nodes], dtype=np.int32)

    @property
    def n_traffic(self):
        return np.array([node.n_traffic for node in self.nodes], dtype=np.int32)

    @property
    def n_transmit(self):
        return np.array([node.is_transmitting for node in self.nodes], dtype=np.int32)

    @property
    def n_drop(self):
        return np.array([node.is_dropping for node in self.nodes], dtype=np.int32)

    @property
    def queue_length(self):
        return np.array([node.queue_length for node in self.nodes], dtype=np.int32)

    @property
    def delay(self):
        return np.array([node.delay for node in self.nodes], dtype=np.int32) # slots

    @property
    def peak_delay(self):
        return np.array([node.peak_delay for node in self.nodes], dtype=np.int32)

    @property
    def cw(self):
        return np.array([node.cw for node in self.nodes], dtype=np.int32)

    @property
    def spatial_p(self):
        return np.array([node.traffic_model.spatial_p(self.now) for node in self.nodes], dtype=np.float32)

    @property
    def busy(self):
        busy_count = np.zeros(self.args.n_rb)
        if self.action is not None:
            for i, rb in enumerate(self.action):
                if self.nodes[i].is_transmitting:
                    busy_count[rb] += 1
        is_busy = (busy_count > 0).astype(np.float32)
        return is_busy
    # =====================================================================================
