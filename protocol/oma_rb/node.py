import numpy as np

class Node:

    def __init__(self, net):
        # save data
        self.net = net
        self.reset()

    # =====================================================================================
    # call by environment
    # =====================================================================================
    def reset(self):
        # drl state
        # ...
        # resource block
        self.rb = 0
        # BACKOFF
        self.cw = 1 # np.ones(self.net.args.n_rb)
        # transmission state
        self.transmit_queue = []
        self.is_transmitting = False
        self.is_dropping = False
        self.n_retry = 0
        self.count = 0
        # arrival state
        self.n_traffic = 0
        # compute peak delay
        self._peak_delay = 0.0
        self._n_success = 0

    def set_rb(self, rb):
        self.rb = rb

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
        # BACKOFF
        if self.cw > 1 and self.count < 1 and self.n_traffic > 0:
            self.count = np.random.randint(low=0, high=self.cw)

    def transmit(self):
        if len(self.transmit_queue) > 0:
            if self.count <= 0:
                self.is_transmitting = True
            else:
                self.count -= 1
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
                # *2 CW
                self.cw = np.min([self.args.max_cw, self.cw * 2])
                self.n_retry += 1
                # new back-off time
                self.count = np.random.randint(low=0, high=self.cw)
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
                # /2 CW
                self.cw = np.max([1, self.cw / 2])

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
