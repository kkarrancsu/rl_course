from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch, vpg_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--algo', type=str, default='vpg',
                        choices=['vpg', 'ppo'])
    args = parser.parse_args()

    eg = ExperimentGrid(name=args.algo+'-pyt-bench')
    eg.add('env_name', 'sorty:sorty-v0', 'n=6', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    if args.algo == 'vpg':
        eg.run(vpg_pytorch, num_cpu=args.cpu)
    elif args.algo == 'ppo':
        eg.run(ppo_pytorch, num_cpu=args.cpu)
    else:
        raise ValueError("Unsupported Training Algorithm!")
