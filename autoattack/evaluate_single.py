import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from robustbench import load_model
from robustbench.data import load_cifar10, load_cifar100

sys.path.insert(1, "./")
from model import create_model, Normalize, PreActResNet18
from spgd import SparsePGD
from rs_attacks import RSAttack
from imagenet100 import imagenet100
from gtsrb import load_gtsrb
from resnet import ResNet34


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test')
parser.add_argument('--data_dir', type=str, default='/data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet100', 'gtsrb'])
parser.add_argument('--ckpt', type=str,
                    help='For robustbench, it is the name. For others, it is the path of checkpoint')
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.25, help='real alpha = alpha * eps')
parser.add_argument('--beta', type=float, default=0.25, help='real beta = beta * sqrt(h*w)')
parser.add_argument('--n_iters', type=int, default=10000)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--n_examples', type=int, default=10000)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--model', type=str, choices=['standard', 'l1', 'l2', 'linf', 'l0'], default='l1')
parser.add_argument('--unprojected', action='store_true')
parser.add_argument('--projected', action='store_true')
parser.add_argument('--black', action='store_true')
parser.add_argument('--calc_aa', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--verbose_interval', type=int, default=100)
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
elif args.dataset == 'imagenet100':
    train_loader, valid_loader, test_loader, num_classes = imagenet100(data_dir=args.data_dir, batch_size=args.bs)
    loader = test_loader
    dataset = loader.dataset
    num_classes = 100
elif args.dataset == 'gtsrb':
    train_loader, test_loader, num_classes, img_size, train_set, test_set = load_gtsrb(
        args.data_dir, args.bs,
        test_batch_size=args.bs,
        eval_samples=args.n_examples,
        num_workers=4)
    loader = test_loader
    dataset = test_set
    num_classes = 43
else:
    raise NotImplementedError

# standard model
if args.model == 'standard':
    if args.dataset != 'imagenet100' and args.dataset != 'gtsrb':
        net = create_model(name='resnet18', num_classes=num_classes, channel=3, norm='batchnorm')
        net.load_state_dict(torch.load(args.ckpt)['state_dict'])
        net = nn.Sequential(
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
            net
        )
    else:
        net = create_model(name='resnet34', num_classes=num_classes, channel=3)
        net.load_state_dict(torch.load(args.ckpt)['state_dict'])
        # net = nn.Sequential(
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     net
        # )
# robust model
elif args.model == 'l2':
    net = load_model(args.ckpt, dataset='cifar10', threat_model='L2')
elif args.model == 'linf':
    net = load_model(args.ckpt, dataset='cifar10', threat_model='Linf')
elif args.model == 'l1':
    if args.dataset != 'imagenet100':
        net = PreActResNet18(n_cls=num_classes, activation='softplus1')
        ckpt2load = torch.load(args.ckpt)
        ckpt = {}
        for k, v in ckpt2load.items():
            ckpt[k[2:]] = v
        net.load_state_dict(ckpt)
    else:
        net = ResNet34(num_classes=num_classes)
        net = nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            net
        )
        ckpt2load = torch.load(args.ckpt)
        net.load_state_dict(ckpt2load)

elif args.model == 'l0':
    if args.dataset != 'imagenet100' and args.dataset != 'gtsrb':
        net = PreActResNet18(num_classes=num_classes, activation='softplus1')
    else:
        net = create_model(name='resnet34', channel=3, num_classes=num_classes)
    # net = nn.Sequential(
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     net
    # )
    try:
        net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    except:
        net.load_state_dict(torch.load(args.ckpt))
else:
    raise NotImplementedError

net = net.cuda()
net.eval()

if args.calc_aa:
    all_ind_fail = None
    all_ind_fail_list = None

if args.unprojected:
    print('Unprojected gradient')
    clean_acc = 0.
    robust_acc = 0.
    avg_it = 0.
    it_list_unproj = np.array([])
    acc_list = [0] * (args.n_iters // args.verbose_interval)
    ind_fails = np.array([])
    ind_fail_list = [np.array([])] * (args.n_iters // args.verbose_interval)
    attacker = SparsePGD(net, epsilon=args.eps, k=args.k, t=args.n_iters, unprojected_gradient=True,
                         patience=args.patience, alpha=args.alpha, beta=args.beta, verbose=True)
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = x.cuda()
        y = y.cuda()
        x_adv, acc, it, acc_list_batch, ind_fail_list_batch = attacker.perturb(x, y)
        with torch.no_grad():
            fool_label = torch.argmax(net(x_adv), dim=1)
            clean_label = torch.argmax(net(x), dim=1)
            clean_acc += (clean_label == y).float().sum().item()
            robust_acc += (fool_label == y).float().sum().item()
            avg_it += it[fool_label != y].sum().item()
            it_list_unproj = np.concatenate((it_list_unproj, it[fool_label != y].cpu().numpy()))
            acc_list = [a+b for a, b in zip(acc_list, acc_list_batch)]
            ind_fail = (fool_label == y).nonzero().squeeze().cpu().numpy()
            if ind_fail.size > 0:
                if ind_fail.ndim == 0:
                    ind_fail = np.expand_dims(ind_fail, axis=0)
                ind_fails = np.concatenate((ind_fails, ind_fail + args.bs * i))
            for j in range(len(ind_fail_list_batch)):
                if ind_fail_list_batch[j].size > 0:
                    if ind_fail_list_batch[j].ndim == 0:
                        ind_fail_list_batch[j] = np.expand_dims(ind_fail_list_batch[j], axis=0)
                    ind_fail_list[j] = np.concatenate((ind_fail_list[j], ind_fail_list_batch[j] + args.bs * i))
    if args.calc_aa:
        all_ind_fail = ind_fails if all_ind_fail is None else np.intersect1d(all_ind_fail, ind_fails)
        all_ind_fail_list = ind_fail_list if all_ind_fail_list is None else [np.intersect1d(a, b) for a, b in zip(all_ind_fail_list, ind_fail_list)]
    avg_it = round(avg_it / (len(dataset) - robust_acc), 2)
    clean_acc = round(clean_acc / len(dataset), 4)
    robust_acc = round(robust_acc / len(dataset), 4)
    acc_list = [round(a / len(dataset), 4) for a in acc_list]
    print('Clean Acc:', clean_acc)
    print('Robust Acc:', robust_acc)
    print('Avg It:', avg_it)
    print('Acc List:', acc_list)
    np.save(os.path.join(save_path, 'it_list_unproj.npy'), it_list_unproj)

if args.projected:
    print('Projected gradient')
    clean_acc = 0.
    robust_acc = 0.
    avg_it = 0.
    it_list_proj = np.array([])
    acc_list = [0] * (args.n_iters // args.verbose_interval)
    ind_fails = np.array([])
    ind_fail_list = [np.array([])] * (args.n_iters // args.verbose_interval)
    attacker = SparsePGD(net, epsilon=args.eps, k=args.k, t=args.n_iters, unprojected_gradient=False,
                         patience=args.patience, alpha=args.alpha, beta=args.beta, verbose=True)
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = x.cuda()
        y = y.cuda()
        x_adv, acc, it, acc_list_batch, ind_fail_list_batch = attacker.perturb(x, y)
        with torch.no_grad():
            fool_label = torch.argmax(net(x_adv), dim=1)
            clean_label = torch.argmax(net(x), dim=1)
            clean_acc += (clean_label == y).float().sum().item()
            robust_acc += (fool_label == y).float().sum().item()
            avg_it += it[fool_label != y].sum().item()
            it_list_proj = np.concatenate((it_list_proj, it[fool_label != y].cpu().numpy()))
            acc_list = [a + b for a, b in zip(acc_list, acc_list_batch)]
            ind_fail = (fool_label == y).nonzero().squeeze().cpu().numpy()
            if ind_fail.size > 0:
                if ind_fail.ndim == 0:
                    ind_fail = np.expand_dims(ind_fail, axis=0)
                ind_fails = np.concatenate((ind_fails, ind_fail + args.bs * i))
            for j in range(len(ind_fail_list_batch)):
                if ind_fail_list_batch[j].size > 0:
                    if ind_fail_list_batch[j].ndim == 0:
                        ind_fail_list_batch[j] = np.expand_dims(ind_fail_list_batch[j], axis=0)
                    ind_fail_list[j] = np.concatenate((ind_fail_list[j], ind_fail_list_batch[j] + args.bs * i))

    if args.calc_aa:
        all_ind_fail = ind_fails if all_ind_fail is None else np.intersect1d(all_ind_fail, ind_fails)
        all_ind_fail_list = ind_fail_list if all_ind_fail_list is None else [np.intersect1d(a, b) for a, b in zip(all_ind_fail_list, ind_fail_list)]
    avg_it = round(avg_it / (len(dataset) - robust_acc), 2)
    clean_acc = round(clean_acc / len(dataset), 4)
    robust_acc = round(robust_acc / len(dataset), 4)
    acc_list = [round(a / len(dataset), 4) for a in acc_list]
    print('Clean Acc:', clean_acc)
    print('Robust Acc:', robust_acc)
    print('Avg It:', avg_it)
    print('Acc List:', acc_list)
    np.save(os.path.join(save_path, 'it_list_proj.npy'), it_list_proj)

if args.black:
    clean_acc = 0.
    robust_acc = 0.
    avg_it = 0.
    it_list_black = np.array([])
    acc_list = [0] * (args.n_iters // args.verbose_interval)
    ind_fails = np.array([])
    ind_fail_list = [np.array([])] * (args.n_iters // args.verbose_interval)
    print('Black-box attack')
    attack = RSAttack(net, norm='L0', eps=args.k, verbose=True, n_queries=args.n_iters, targeted=False,
                      p_init=0.3 if args.model == 'standard' else 0.8, loss='margin' if args.model == 'standard' else 'ce')
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
            x = x.cuda()
            y = y.cuda()
            qr_curr, adv, acc_list_batch, ind_fail_list_batch = attack.perturb(x, y)
            output = net(adv.cuda())
            a = (output.max(1)[1] == y)
            b = (net(x).max(1)[1] == y)
            robust_acc += a.float().sum().item()
            clean_acc += b.float().float().sum().item()
            avg_it += qr_curr[~a].sum().item()
            it_list_black = np.concatenate((it_list_black, qr_curr[~a].cpu().numpy()))
            acc_list = [a + b for a, b in zip(acc_list, acc_list_batch)]
            ind_fail = a.nonzero().squeeze().cpu().numpy()
            if ind_fail.size > 0:
                if ind_fail.ndim == 0:
                    ind_fail = np.expand_dims(ind_fail, axis=0)
                ind_fails = np.concatenate((ind_fails, ind_fail + args.bs * i))
            for j in range(len(ind_fail_list_batch)):
                if ind_fail_list_batch[j].size > 0:
                    if ind_fail_list_batch[j].ndim == 0:
                        ind_fail_list_batch[j] = np.expand_dims(ind_fail_list_batch[j], axis=0)
                    ind_fail_list[j] = np.concatenate((ind_fail_list[j], ind_fail_list_batch[j] + args.bs * i))
    if args.calc_aa:
        all_ind_fail = ind_fails if all_ind_fail is None else np.intersect1d(all_ind_fail, ind_fails)
        all_ind_fail_list = ind_fail_list if all_ind_fail_list is None else [np.intersect1d(a, b) for a, b in zip(all_ind_fail_list, ind_fail_list)]
    avg_it = round(avg_it / (len(dataset) - robust_acc), 2)
    clean_acc = round(clean_acc / len(dataset), 4)
    robust_acc = round(robust_acc / len(dataset), 4)
    acc_list = [round(len(a) / len(dataset), 4) for a in ind_fail_list]
    print('Clean Acc:', clean_acc)
    print('Robust Acc:', robust_acc)
    print('Avg It:', avg_it)
    print('Acc List:', acc_list)
    np.save(os.path.join(save_path, 'it_list_black.npy'), it_list_black)

if args.calc_aa:
    print('Robust Acc of AA:', round(len(all_ind_fail) / len(dataset), 4))
    print('Acc List of AA:', [round(len(a) / len(dataset), 4) for a in all_ind_fail_list])

