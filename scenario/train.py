import numpy as np

import algorithm
import protocol

def train(args):
    # create SCMA environment
    env = protocol.Env(args)
    states = env.reset()
    # create monitor
    monitor = protocol.Monitor(env, args)
    # create controller
    controller = algorithm.create_controller(env, args)
    controller = controller.load_best(env, monitor)
    # run
    try:
        controller.run(monitor)
    # save
    except KeyboardInterrupt:
        controller.save(monitor)
    controller.save(monitor)
