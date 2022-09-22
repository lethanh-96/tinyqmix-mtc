import algorithm
import protocol

def debug(args):
    # create environment
    env = protocol.Env(args)
    # create MAC controller
    controller = algorithm.create_controller(env, args)
    # create monitor    
    args.n_step = args.n_test_step
    monitor = protocol.Monitor(env, args)
    # run test
    state = env.reset()

    print(state)

    # for i in range(args.n_step * args.n_test_episode):
    #     if i % args.n_step == 0:
            # state = env.reset()
        # action = controller.select_np_action(state)
