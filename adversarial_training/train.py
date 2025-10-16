from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import pprint
from time import gmtime, strftime
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from Logging import Logger
from config import cfg
from model import create_model, Normalize
from trainer import Trainer, TrainerSAT
from evaluator import Evaluator
from utils import *
from spgd import SparsePGD, FastSparsePGD


if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    if not cfg.randomize:
        # set fixed seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
    log_path = os.path.join(cfg.log_dir, cfg.exp_name)
    mkdir_if_missing(log_path)
    snap_path = os.path.join(cfg.snap_dir, cfg.exp_name)
    mkdir_if_missing(snap_path)

    summary_writer = None
    if not cfg.no_log:
        log_name = cfg.exp_name + "_log_" + \
                   strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.txt'
        sys.stdout = Logger(os.path.join(log_path, log_name))
        summary_writer = SummaryWriter(log_dir=log_path)

    print("Input Args: ")
    pprint.pprint(cfg)
    train_loader, test_loader, num_classes, img_size, train_set, test_set = get_data_loader(
        cfg,
        data_name=cfg.data_name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.eval_batch_size,
        eval_samples=cfg.eval_samples,
        num_workers=4)

    model = create_model(name=cfg.model_name, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=5e-4, momentum=0.9)

    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if torch.cuda.is_available():
            for k, v in optimizer.state.items():
                if 'momentum_buffer' not in v:
                    continue
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'].cuda()
        print('Resume from epoch {}'.format(start_epoch))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[
            cfg.max_epoch * len(train_loader) // 4,
            3 * cfg.max_epoch * len(train_loader) // 4,
        ], gamma=0.1, last_epoch=start_epoch - 1)

    is_cuda = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
        is_cuda = True

    if cfg.n_iters > 1:
        attack = SparsePGD(model, epsilon=cfg.epsilon, k=cfg.k, t=cfg.n_iters, alpha=cfg.alpha, beta=cfg.beta,
                           nfgsm=cfg.nfgsm, patience=10)
    else:
        attack = FastSparsePGD(model, epsilon=cfg.epsilon, k=cfg.k, t=cfg.n_iters, alpha=cfg.alpha, beta=cfg.beta,
                               nfgsm=cfg.nfgsm,
                               loss=cfg.attack_loss)
    # attack_eval = SparsePGD(model, epsilon=cfg.epsilon, k=cfg.k, t=cfg.n_iters, alpha=cfg.alpha, beta=cfg.beta)

    if cfg.sat:
        trainer = TrainerSAT(model=model, attack=attack, optimizer=optimizer,
                         summary_writer=summary_writer, is_cuda=True,
                         output_freq=cfg.output_freq, print_freq=cfg.print_freq,
                         base_lr=cfg.lr, max_epoch=cfg.max_epoch,
                         steps=cfg.steps, rate=cfg.decay_rate, loss=cfg.train_loss, trades_beta=cfg.trades_beta,
                         scheduler=scheduler, sat_epoch=int(cfg.max_epoch * cfg.sat_epoch))
    else:
        trainer = Trainer(model=model, attack=attack, optimizer=optimizer,
                             summary_writer=summary_writer, is_cuda=True,
                             output_freq=cfg.output_freq, print_freq=cfg.print_freq,
                             base_lr=cfg.lr, max_epoch=cfg.max_epoch,
                             steps=cfg.steps, rate=cfg.decay_rate, loss=cfg.train_loss, trades_beta=cfg.trades_beta,
                             scheduler=scheduler)
    trainer.ready_data(train_set, num_classes)
    # evaluator = Evaluator(model=model, attack=attack_eval, is_cuda=is_cuda, verbose=True)

    trainer.reset()
    for epoch in range(start_epoch, cfg.max_epoch):
        if cfg.train_mode == 'alter':
            if (epoch+1) % 5 == 0:
                attack.change_masking()

        trainer.train(epoch, train_loader)

        if cfg.save_freq < 1:
            save_current = False
        else:
            save_current = (epoch + 1) % cfg.save_freq == 0 \
                           or epoch == 0 or epoch == cfg.max_epoch - 1
        if save_current:
            # nat_acc, adv_acc = evaluator.evaluate(test_loader)
            # if summary_writer is not None:
            #     summary_writer.add_scalar('nat_acc', nat_acc, epoch)
            #     summary_writer.add_scalar('adv_acc', adv_acc, epoch)
            # print("epoch {:3d} evaluated".format(epoch))
            # print("natural     accuracy: {:.4f}".format(nat_acc))
            # print("adversarial accuracy: {:.4f}".format(adv_acc))
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            dict_to_save = {'state_dict': state_dict,
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch + 1}
            fpath = os.path.join(snap_path, 'checkpoint_' +
                                 str(epoch + 1) + '.pth')
            torch.save(dict_to_save, fpath)

    trainer.close()
