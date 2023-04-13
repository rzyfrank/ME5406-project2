import argparse
import random

def obtain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Train')
    parser.add_argument('--algorithm', type=str, default='DuelingDQN')
    parser.add_argument('--track_h', type=int, default=1300)
    parser.add_argument('--track_w', type=int, default=2100)
    parser.add_argument('--track_width', type=int, default=100)
    parser.add_argument('--car_path', type=str, default='./car.png')
    parser.add_argument('--top_view_size', type=int, default=350)
    parser.add_argument('--front_view_w', type=int, default=224)
    parser.add_argument('--front_view_h', type=int, default=224)
    parser.add_argument('--split', type=int, default=100)
    parser.add_argument('--in_channel', type=int, default=3)

    parser.add_argument('--episode', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--resume_checkpoint', type=str)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--rand_seed', type=int)
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    return args
