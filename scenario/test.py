import algorithm
import protocol
import torch

CHANNEL_COHERENCE = 50

def add_action_noise(i, controller, action, n_action, eps=0.05):
    ra = torch.randint(low=0, 
                       high=n_action, 
                       size=(controller.args.n_node,), 
                       device=controller.args.device)
    r   = torch.rand(controller.args.n_node)
    if i < controller.args.n_step:
        idx = r < 1
    else:
        idx = r < eps
    action[idx] = ra[idx]
    return action

def add_action_noise_v2(i, controller, action, qs, eps=0.01):
    second_best_action = torch.topk(qs, 2, dim=1)[1][:, 1]
    r = torch.rand(controller.args.n_node)
    idx = r < eps
    action[idx] = second_best_action[idx]
    return action

def select_np_action(i, controller, states):
    if controller.args.algorithm != 'qmix_lstm':
        if controller.args.mode == 'train':
            states = controller.preprocessor.fit_transform(states)
        else:
            states = controller.preprocessor.transform(states)
    qs = controller.mixer.compute_policy_qs(states)
    actions = qs.max(1)[1]
    actions = add_action_noise_v2(i, controller, actions, qs)
    # actions = add_action_noise(i, controller, actions, controller.args.n_rb)
    return actions.detach().cpu().numpy()

def test(args):
    # create environment
    env = protocol.Env(args)
    # create MAC controller
    controller = algorithm.create_controller(env, args)
    # load trained controller if algorithm is DRL
    if 'qmix' in args.algorithm:
        args.n_step = args.n_train_step
        monitor = protocol.Monitor(env, args)
        # controller = controller.load(env, monitor)
        controller = controller.load_best(env, monitor)

    # create monitor    
    args.n_step = args.n_test_step
    monitor = protocol.Monitor(env, args)
    # run test
    state = env.reset()
    states = []
    for i in range(args.n_step * args.n_test_episode):
        if i % args.n_step == 0:
            state = env.reset()
        if 'qmix' in args.algorithm:
            action = select_np_action(i, controller, state)
        else:
            action = controller.select_np_action(state)
        state, reward, done, info = env.step(action)
        if 'dql' in args.algorithm:
            controller.update(action, reward.item())
        monitor.step(info)
