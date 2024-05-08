import datetime
import argparse
import os
import sys
import torch
import torch.nn as nn
from robustbench import load_model

sys.path.insert(1, "./")
from model import create_model, Normalize, PreActResNet18
from resnet import ResNet34
from imagenet100 import imagenet100
from gtsrb import load_gtsrb
from saa_large import SparseAutoAttack

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'gtsrb'])
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

if args.dataset == 'imagenet100':
    train_loader, valid_loader, test_loader, num_classes = imagenet100(data_dir=args.data_dir, batch_size=args.bs, out_index=True)
    loader = test_loader
    num_classes = 100

elif args.dataset == 'gtsrb':
    train_loader, test_loader, num_classes, img_size, train_set, test_set = load_gtsrb(
        args.data_dir, args.bs,
        test_batch_size=args.bs,
        eval_samples=args.n_examples,
        num_workers=4, out_index=True)
    loader = test_loader
    dataset = test_set
    num_classes = 43

# standard model
if args.model == 'standard':
    net = create_model(name='resnet34', num_classes=num_classes, channel=3)
    net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    # net = nn.Sequential(
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     net
    # )

# robust model
elif args.model == 'l2':
    net = load_model(args.ckpt, dataset=args.dataset, threat_model='L2')
elif args.model == 'linf':
    net = load_model(args.ckpt, dataset=args.dataset, threat_model='Linf')
elif args.model == 'l1':
    net = ResNet34(num_classes=num_classes)
    net = nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        net
    )
    ckpt2load = torch.load(args.ckpt)
    # ckpt = {}
    # for k, v in ckpt2load.items():
    #     ckpt[k[2:]] = v
    net.load_state_dict(ckpt2load)
elif args.model == 'l0':
    net = create_model(name='resnet34', channel=3, num_classes=num_classes)
    # net = nn.Sequential(
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     net
    # )
    ckpt2load = torch.load(args.ckpt)["state_dict"]
    net.load_state_dict(ckpt2load)
else:
    raise NotImplementedError

net = net.cuda()
net.eval()
attacker = SparseAutoAttack(net, args, max_candidates=10)
clean_acc, black_robust_acc, time_used = attacker(loader, dataset=loader.dataset, args=args)
print('Time used:', datetime.timedelta(seconds=time_used))
