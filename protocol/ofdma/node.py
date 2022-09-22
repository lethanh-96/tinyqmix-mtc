import numpy as np

class Node:

    def __init__(self, net):
        self.net = net
        self.reset()

    # =====================================================================================
    # call by environment
    # =====================================================================================
    def reset(self):
        self.schedule = None
        # BACKOFF
        self.cw = 1 # np.ones(self.net.args.n_rb)
        # transmission state
        self.transmit_queue = []
        self.is_transmitting = False
        self.is_dropping = False
        self.n_retry = 0
        # arrival state
        self.n_traffic = 0
        # compute peak delay
        self._peak_delay = 0.0
        self._n_success = 0

    def set_schedule(self, schedule):
        self.schedule = schedule

    def set_arrival_model(self, model):
        self.arrival_model = model
    # =====================================================================================

    # =====================================================================================
    # call by network
    # =====================================================================================
    def generate_traffic(self):
        # extract param
        # get number of new packet
        self.n_traffic = self.traffic_model.n_packet(self.now)
        # send enqueue new packet
        for _ in range(self.n_traffic):
            if len(self.transmit_queue) < self.args.queue_length:
                self.transmit_queue.append(self.now)

    def transmit(self):
        if len(self.transmit_queue) > 0:
            # if the schedule allow sending a packet at
            # any ctu
            if np.sum(self.schedule[self.now % self.args.coherrent_time]) > 0:
                self.is_transmitting = True
            else:
                self.is_transmitting = False
        else:
            self.is_transmitting = False
        return self.is_transmitting

    def ack(self, collision):
        # INITIALIZE
        self.is_dropping = False
        # 
        if self.is_transmitting:
            # COLLISION
            if collision: 
                self.n_retry += 1
                # DROP
                if self.n_retry > self.args.max_retry:
                    self.transmit_queue.pop(0)
                    self.is_dropping = True
                    self.n_retry = 0
                else:
                    self.n_retry += 1
            else:
                # SUCCESS
                # record peak delay
                self._peak_delay += self.delay
                self._n_success  += 1
                # dequeue packet
                self.transmit_queue.pop(0)
                self.n_retry = 0

    def clean(self):
        self._peak_delay = 0.0
        self._n_success  = 0
    # =====================================================================================

    # =====================================================================================
    # other properties
    # =====================================================================================
    @property
    def args(self):
        return self.net.args

    @property
    def now(self):
        return self.net.now

    @property
    def delay(self):
        if len(self.transmit_queue) > 0:
            return float(self.net.now - self.transmit_queue[0])
        else:
            return 0.0

    @property
    def queue_length(self):
        return len(self.transmit_queue)

    @property
    def peak_delay(self):
        if self._n_success > 0:
            return self._peak_delay / self._n_success
        else:
            return 0
    # =====================================================================================
