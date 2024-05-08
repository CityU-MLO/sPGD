from torch.nn import functional as F
import torch
import sys

from tqdm import tqdm
import time
from spgd import SparsePGD
from torch.utils.data import DataLoader, TensorDataset

# sys.path.insert(1, "./../")
from rs_attacks import RSAttack


class SparseAutoAttack(object):

    def __init__(self, model, args, black_att='RS', black_iters=10000, p_init=0.8, max_candidates=10):
        self.model = model
        self.att1 = SparsePGD(model, epsilon=args.eps, k=args.k, t=args.n_iters,
                              unprojected_gradient=args.unprojected_gradient)
        self.att2 = SparsePGD(model, epsilon=args.eps, k=args.k, t=args.n_iters,
                              unprojected_gradient=not args.unprojected_gradient)
        if black_att == 'RS':
            self.black = RSAttack(model, norm='L0', eps=args.k, verbose=False, n_queries=black_iters, p_init=p_init,
                                  targeted=False, loss='ce')
        else:
            raise NotImplementedError('Black-box attack {} not implemented'.format(black_att))
        self.k = args.k
        self.max_candidates = max_candidates

    def untarget_attack(self, att, loader, clean=False):
        x_next, y_next = None, None
        if clean:
            clean_acc = 0.

        untarget_robust_acc = 0.

        for x, y in tqdm(loader):
            x = x.cuda()
            y = y.cuda()

            original_y = y.clone()
            # clean results
            clean_label = att.model(x).argmax(dim=1)
            fool_label = clean_label.clone()
            c_acc = (clean_label == y).float().sum().item()
            ind_fail = (clean_label == y).nonzero().squeeze()
            x_fail, y_fail = x[ind_fail], y[ind_fail]
            if x_fail.dim() == 3:
                x_fail = x_fail.unsqueeze(0)
                y_fail = y_fail.unsqueeze(0)

            unt_r_acc = 0.
            # untargeted attack
            if ind_fail.numel() > 0:
                x_adv, acc, i = att(x_fail, y_fail, targeted=False)
                output = att.model(x_adv)
                pred = output.argmax(dim=1)
                # pred = att.model(x + self.masking.apply(perturb, mask, self.k)).argmax(dim=1)
                ind_untargeted_success = (pred != y_fail).nonzero().squeeze()
                if ind_untargeted_success.numel() > 0:
                    if ind_fail.numel() == 1:
                        fool_label[ind_fail] = pred[ind_untargeted_success]
                    else:
                        fool_label[ind_fail[ind_untargeted_success]] = pred[ind_untargeted_success]
                unt_r_acc = (fool_label == original_y).float().sum().item()
                ind_untargeted_fail = (pred == y_fail).nonzero().squeeze()
                try:
                    if ind_fail.dim() == 0:
                        ind_fail = ind_fail.unsqueeze(0)
                    ind_fail = ind_fail[ind_untargeted_fail]
                except:
                    raise ValueError('ind_fail: ', ind_fail, 'ind_untargeted_fail: ', ind_untargeted_fail)

            if ind_fail.numel() > 0:
                # x_fail, y_fail = x_fail[ind_fail], y_fail[ind_fail]
                x_fail, y_fail = x[ind_fail], y[ind_fail]
                if x_fail.dim() == 3:
                    x_fail = x_fail.unsqueeze(0)
                    y_fail = y_fail.unsqueeze(0)
                x_next = x_fail if x_next is None else torch.cat((x_next, x_fail), dim=0)
                y_next = y_fail if y_next is None else torch.cat((y_next, y_fail), dim=0)
            if clean:
                clean_acc += c_acc
            untarget_robust_acc += unt_r_acc
        if clean:
            return x_next, y_next, clean_acc, untarget_robust_acc
        else:
            return x_next, y_next, untarget_robust_acc

    def target_attack(self, att, loader):
        x_next, y_next = None, None

        target_robust_acc = 0.

        for x, y in tqdm(loader):
            x = x.cuda()
            y = y.cuda()
            original_y = y.clone()

            # perturb, mask, acc, _ = att(x, y, targeted=False)
            # output = att.model(x + att.masking.apply(perturb, F.sigmoid(mask), self.k))
            output = att.model(x)

            clean_label = att.model(x).argmax(dim=1)
            fool_label = clean_label.clone()

            class_candidates = torch.topk(output, self.max_candidates, dim=-1)[1].to(y.device)
            # class_candidates = torch.arange(self.classes).expand(x.size(0), self.classes).to(y.device)
            class_candidates = class_candidates[(class_candidates != y.unsqueeze(1))].view(x.size(0), -1).permute(1, 0)
            assert class_candidates.shape == (self.max_candidates - 1, x.size(0)), 'class candidates shape error'
            ind_fail2 = torch.arange(x.size(0)).to(x.device)  # index on current x
            # ind_fail: index on the original x

            for i in range(class_candidates.size(0)):
                target_class = class_candidates[i, ind_fail2]
                if target_class.dim() == 0:
                    target_class = target_class.unsqueeze(0)
                x_adv, acc, _ = att(x, y, targeted=True, target=target_class)
                pred = att.model(x_adv).argmax(dim=1)
                # pred = att.model(x + self.masking.apply(perturb, mask, self.k)).argmax(dim=1)
                ind_targeted_success = (pred == target_class).nonzero().squeeze()
                if ind_targeted_success.numel() > 0:
                    fool_label[ind_targeted_success] = pred[ind_targeted_success]
                    # if ind_fail.numel() == 1:
                    #     fool_label[ind_fail] = pred[ind_targeted_success]
                    # else:
                    #     fool_label[ind_fail[ind_targeted_success]] = pred[ind_targeted_success]

                ind_targeted_fail = (pred != target_class).nonzero().squeeze()
                # ind_fail = ind_fail[ind_targeted_fail] if ind_fail.numel() > 1 else ind_fail
                ind_fail2 = ind_fail2[ind_targeted_fail] if ind_fail2.numel() > 1 else ind_fail2
                if ind_targeted_fail.numel() == 0:
                    break
                x, y = x[ind_targeted_fail], y[ind_targeted_fail]
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)
            x_next = x if x_next is None else torch.cat((x_next, x), dim=0)
            y_next = y if y_next is None else torch.cat((y_next, y), dim=0)

            target_robust_acc += (fool_label == original_y).float().sum().item()
        return x_next, y_next, target_robust_acc

    def black_attack(self, att, loader):
        robust_acc_for_black = 0.
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.cuda()
                y = y.cuda()
                # st = time.time()
                qr_curr, adv = att.perturb(x, y)
                # et = time.time()
                # time_used += et - st
                output = self.model(adv.cuda())
                robust_acc_for_black += (output.max(1)[1] == y).float().sum().item()
                ind_succ = (output.max(1)[1] != y).nonzero().squeeze()

                ind_fail = (output.max(1)[1] == y).nonzero().squeeze()
                if ind_fail.numel() > 0:
                    x_fail, y_fail = x[ind_fail], y[ind_fail]
                    if x_fail.dim() == 3:
                        x_fail = x_fail.unsqueeze(0)
                        y_fail = y_fail.unsqueeze(0)

        # robust_acc = round(robust_acc * robust_acc_for_black / len(dataset_for_black), 4)
        robust_acc = robust_acc_for_black
        return robust_acc

    def __call__(self, loader):
        ###################################
        #### Untargeted Attack 1 ##########
        ###################################
        time_used = 0.
        print('Untargeted Attack 1')
        st = time.time()
        x_for_unt_2, y_for_unt_2, clean_acc, untarget_robust_acc_1 = self.untarget_attack(self.att1, loader, clean=True)
        ed = time.time()
        time_used += ed - st
        print('Clean accuracy: {:.2f}%'.format(clean_acc / len(loader.dataset) * 100))
        print('Untargeted robust accuracy: {:.2f}%'.format(untarget_robust_acc_1 / len(loader.dataset) * 100))
        if x_for_unt_2 is None:
            return clean_acc, 0., time_used

        ###################################
        #### Untargeted Attack 2 ##########
        ###################################
        print('Untargeted Attack 2')
        x_for_unt_2 = x_for_unt_2.cpu()
        y_for_unt_2 = y_for_unt_2.cpu()

        dataset_for_unt_2 = TensorDataset(x_for_unt_2, y_for_unt_2)
        print('# White-box 1 untargeted attack failed samples:', len(dataset_for_unt_2))
        loader_for_unt_2 = DataLoader(dataset_for_unt_2,
                                      batch_size=loader.batch_size,
                                      shuffle=False,
                                      num_workers=2)
        st = time.time()
        # x_for_tar_1, y_for_tar_1, untarget_robust_acc_2 = self.untarget_attack(self.att2, loader=loader_for_unt_2,
        #                                                                        clean=False)
        x_for_black, y_for_black, untarget_robust_acc_2 = self.untarget_attack(self.att2, loader=loader_for_unt_2,
                                                                               clean=False)
        ed = time.time()
        time_used += ed - st
        print('Untargeted robust accuracy: {:.2f}%'.format(untarget_robust_acc_2 / len(loader.dataset) * 100))
        # if x_for_tar_1 is None:
        if x_for_black is None:
            return clean_acc, 0., time_used

        # ###################################
        # #### Targeted Attack 1 ############
        # ###################################
        # print('Targeted Attack 1')
        # x_for_tar_1 = x_for_tar_1.cpu()
        # y_for_tar_1 = y_for_tar_1.cpu()
        #
        # dataset_for_tar_1 = TensorDataset(x_for_tar_1, y_for_tar_1)
        # print('# White-box 2 untargeted attack failed samples:', len(dataset_for_tar_1))
        # loader_for_tar_1 = DataLoader(dataset_for_tar_1,
        #                               batch_size=loader.batch_size,
        #                               shuffle=False,
        #                               num_workers=2)
        # st = time.time()
        # x_for_tar_2, y_for_tar_2, targeted_robust_acc_1 = self.target_attack(self.att1, loader=loader_for_tar_1)
        # ed = time.time()
        # time_used += ed - st
        # print('Targeted robust accuracy: {:.2f}%'.format(targeted_robust_acc_1 / len(loader.dataset) * 100))
        # if x_for_tar_2 is None:
        #     return clean_acc, 0., time_used
        #
        # ###################################
        # #### Targeted Attack 2 ############
        # ###################################
        # print('Targeted Attack 2')
        # x_for_tar_2 = x_for_tar_2.cpu()
        # y_for_tar_2 = y_for_tar_2.cpu()
        #
        # dataset_for_tar_2 = TensorDataset(x_for_tar_2, y_for_tar_2)
        # print('# White-box 1 targeted attack failed samples:', len(dataset_for_tar_2))
        # loader_for_tar_2 = DataLoader(dataset_for_tar_2,
        #                               batch_size=loader.batch_size,
        #                               shuffle=False,
        #                               num_workers=2)
        # st = time.time()
        # x_for_black, y_for_black, targeted_robust_acc_2 = self.target_attack(self.att2, loader=loader_for_tar_2)
        # ed = time.time()
        # time_used += ed - st
        # print('Targeted robust accuracy: {:.2f}%'.format(targeted_robust_acc_2 / len(loader.dataset) * 100))
        # if x_for_black is None:
        #     return clean_acc, 0., time_used####################
        # #### Targeted Attack 1 ############
        # ###################################
        # print('Targeted Attack 1')
        # x_for_tar_1 = x_for_tar_1.cpu()
        # y_for_tar_1 = y_for_tar_1.cpu()
        #
        # dataset_for_tar_1 = TensorDataset(x_for_tar_1, y_for_tar_1)
        # print('# White-box 2 untargeted attack failed samples:', len(dataset_for_tar_1))
        # loader_for_tar_1 = DataLoader(dataset_for_tar_1,
        #                               batch_size=loader.batch_size,
        #                               shuffle=False,
        #                               num_workers=2)
        # st = time.time()
        # x_for_tar_2, y_for_tar_2, targeted_robust_acc_1 = self.target_attack(self.att1, loader=loader_for_tar_1)
        # ed = time.time()
        # time_used += ed - st
        # print('Targeted robust accuracy: {:.2f}%'.format(targeted_robust_acc_1 / len(loader.dataset) * 100))
        # if x_for_tar_2 is None:
        #     return clean_acc, 0., time_used
        #
        # ###################################
        # #### Targeted Attack 2 ############
        # ###################################
        # print('Targeted Attack 2')
        # x_for_tar_2 = x_for_tar_2.cpu()
        # y_for_tar_2 = y_for_tar_2.cpu()
        #
        # dataset_for_tar_2 = TensorDataset(x_for_tar_2, y_for_tar_2)
        # print('# White-box 1 targeted attack failed samples:', len(dataset_for_tar_2))
        # loader_for_tar_2 = DataLoader(dataset_for_tar_2,
        #                               batch_size=loader.batch_size,
        #                               shuffle=False,
        #                               num_workers=2)
        # st = time.time()
        # x_for_black, y_for_black, targeted_robust_acc_2 = self.target_attack(self.att2, loader=loader_for_tar_2)
        # ed = time.time()
        # time_used += ed - st
        # print('Targeted robust accuracy: {:.2f}%'.format(targeted_robust_acc_2 / len(loader.dataset) * 100))
        # if x_for_black is None:
        #     return clean_acc, 0., time_used

        ###################################
        #### Black box attack #############
        ###################################
        print('Black-box Attack using SparseRS')
        x_for_black = x_for_black.cpu()
        y_for_black = y_for_black.cpu()

        dataset_for_black = TensorDataset(x_for_black, y_for_black)
        print('# While-box 2 targeted attack failed samples:', len(dataset_for_black))
        loader_for_black = DataLoader(dataset_for_black,
                                      batch_size=2*loader.batch_size,
                                      shuffle=False,
                                      num_workers=2)
        st = time.time()
        black_robust_acc = self.black_attack(self.black, loader=loader_for_black)
        ed = time.time()
        time_used += ed - st
        print('Final robust accuracy: {:.2f}%'.format(black_robust_acc / len(loader.dataset) * 100))
        return clean_acc, black_robust_acc, time_used
