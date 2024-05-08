import datetime
import sys
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from robustbench import load_model
from robustbench.data import load_cifar10, load_cifar100

sys.path.insert(1, "./")
from model import create_model, Normalize, PreActResNet18
from saa import SparseAutoAttack

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--ckpt', type=str,
                    help='For robustbench, it is the name. For others, it is the path of checkpoint')
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.25, help='real alpha = alpha * eps')
parser.add_argument('--beta', type=float, default=0.25, help='real beta = beta * sqrt(h*w)')
parser.add_argument('--n_iters', type=int, default=300)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--n_examples', type=int, default=10000)
parser.add_argument('--model', type=str, choices=['standard', 'l1', 'l2', 'linf', 'l0'], default='l1')
parser.add_argument('--unprojected_gradient', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)

save_path = os.path.join('exp', f'{args.exp}_{args.model}_k{args.k}_{args.n_examples}examples_{args.n_iters}iters')
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.dataset == 'cifar10':
    x_test, y_test = load_cifar10(n_examples=args.n_examples, data_dir=args.data_dir)
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    num_classes = 10

elif args.dataset == 'cifar100':
    x_test, y_test = load_cifar100(n_examples=args.n_examples, data_dir=args.data_dir)
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    num_classes = 100

# standard model
if args.model == 'standard':
    net = create_model(name='resnet18', num_classes=num_classes, channel=3, norm='batchnorm')
    net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    net = nn.Sequential(
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        net
    )
# robust model
elif args.model == 'l2':
    net = load_model(args.ckpt, dataset=args.dataset, threat_model='L2')
elif args.model == 'linf':
    net = load_model(args.ckpt, dataset=args.dataset, threat_model='Linf')
elif args.model == 'l1':
    net = PreActResNet18(n_cls=num_classes, activation='softplus1')
    ckpt2load = torch.load(args.ckpt)
    ckpt = {}
    for k, v in ckpt2load.items():
        ckpt[k[2:]] = v
    net.load_state_dict(ckpt)
elif args.model == 'l0':
    net = PreActResNet18(n_cls=num_classes, activation='softplus1')
    # net = nn.Sequential(
    #     Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    #     net
    # )
    net.load_state_dict(
        torch.load(args.ckpt)['state_dict'])
else:
    raise NotImplementedError

net = net.cuda()
net.eval()
attacker = SparseAutoAttack(net, args, max_candidates=5 if args.dataset == 'cifar10' else 10)
clean_acc, black_robust_acc, time_used = attacker(loader)
print('Time used:', datetime.timedelta(seconds=time_used))
