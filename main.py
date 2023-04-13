# import gym
import cv2
import torch

from lib import Env, obtain_args
from network import DQNet, DuelingDQNet
from agent import DQAgent, DuelingDQAgent, DoubleDQAgent

import wandb

def main(args, turning_point, action_map):
    print(args.rand_seed)
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    # Use wandb to visualize Data, comment it when testing
    wandb.init(project='me5046',
               name='DQN-training')

    num_action = len(action_map)
    num_data = 2
    # Create the network
    if args.algorithm == 'DQN' or args.algorithm == 'DoubleDQN':
        print('Network: {:}'.format(args.algorithm))
        net = DQNet(args.in_channel, num_action, num_data)
    elif args.algorithm == 'DuelingDQN':
        print('Network: {:}'.format(args.algorithm))
        net = DuelingDQNet(args.in_channel, num_action, num_data)
    else:
        raise NotImplementedError(f'Not this algorithm:{args.algorithm}')

    # Load trained network
    if args.resume_checkpoint is not None:
        net.load_state_dict(torch.load(args.resume_checkpoint))

    net = net.to(device)

    # Create the environment
    env = Env(args, turning_point)

    # Create the Agent
    if args.algorithm == 'DQN':
        print('Agent: {:}'.format(args.algorithm))
        agent = DQAgent(
            args,
            env,
            net,
            device,
            action_map
        )
    elif args.algorithm == 'DoubleDQN':
        print('Agent: {:}'.format(args.algorithm))
        agent = DoubleDQAgent(
            args,
            env,
            net,
            device,
            action_map
        )
    elif args.algorithm == 'DuelingDQN':
        print('Agent: {:}'.format(args.algorithm))
        agent = DuelingDQAgent(
            args,
            env,
            net,
            device,
            action_map
        )
    else:
        raise NotImplementedError(f'Not this algorithm:{args.algorithm}')

    # Start training or testing
    if args.type == 'Train':
        agent.work()
    elif args.type == 'Test':
        a = agent.test()
    else:
        raise NotImplementedError(f'Not this type:{args.type}')

    wandb.finish()




if __name__ == '__main__':
    args = obtain_args()

    # Choose the track
    # turning_point = [
    #     (400, 500),
    #     (550, 500),
    #     (700, 500)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (550, 500),
    #     (700, 500),
    #     (700, 700)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (550, 500),
    #     (700, 500),
    #     (700, 700),
    #     (850, 550)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (550, 500),
    #     (700, 500),
    #     (700, 700),
    #     (850, 550),
    #     (1000, 550)
    # ]
    turning_point = [
        (400, 500),
        (700, 500),
        (700, 800),
        (1000, 500),
        (1300, 800),
        (1600, 800)
    ]
    # turning_point = [
    #     (400, 500),
    #     (700, 500),
    #     (700, 800),
    #     (1000, 800),
    #     (1200, 600),
    #     (1600, 600)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (400, 800),
    #     (600, 600),
    #     (900, 600),
    #     (1100, 800),
    #     (1400, 800)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (700, 500),
    #     (700, 800),
    #     (1000, 800),
    #     (1300, 500),
    #     (1600, 800)
    # ]
    # turning_point = [
    #     (400, 500),
    #     (400, 800),
    #     (700, 500),
    #     (1000, 500),
    #     (1000, 800),
    #     (1300, 800)
    # ]

    #Choose the angle and speed
    angle_map = [angle for angle in range(-15,20,5)]
    speed_map = [20, -10, -5, 5, 10, 15, 20]
    action_map = []
    for speed in speed_map:
        for angle in angle_map:
            action_map.append([angle,speed])
    main(args, turning_point, action_map)



