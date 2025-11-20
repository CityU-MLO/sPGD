# Sparse-PGD

This is the code repository for papers
- [Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation ](https://arxiv.org/abs/2405.05075), TPAMI 2025.
- [Understanding and Improving Fast Adversarial Training against $l_0$ Bounded Perturbations](https://openreview.net/forum?id=YOv9CANvDv), NeurIPS 2025.
- [Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations](https://openreview.net/forum?id=2bUFIsg2f5), ICML 2024.

## Requirements

To execute the code, please make sure that the following packages are installed:

- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [robustbench](https://github.com/RobustBench/robustbench)

## Structured Sparse Perturbation

To generate structured sparse perturbations, please check class **StructureSparsePGD** in [autoattack/spgd.py](autoattack/spgd.py) and [adversarial_training/spgd.py](adversarial_training/spgd.py)

## Fast-LS-$l_0$

This is the implementation of Fast-LS-$l_0$ in "Understanding and Improving Fast Adversarial Training against $l_0$ Bounded Perturbations, NeurIPS 2025."

Run the following command to train PreActResNet18 on adversarial samples:

```
python train.py --exp_name debug --data_name [DATASET NAME] --data_dir [DATA PATH] --model_name preactresnet --max_epoch 100 --batch_size 128 --lr 0.05 --train_loss [adv, trades] -k 120 --alpha 1 --beta 1 --n_iters 1 --nfgsm --sat --sat_epoch 0.5 --attack_loss ce --gpu 0 
```

- exp_name: experiment name
- data_name: choose from cifar10, cifar100, imagenet100 or gtsrb 
- data_dir: path to the dataset
- model_name: choose a model for training, e.g., 'preactresnet' or 'resnet34'
- max_epoch: number of epochs for training
- batch_size: batch size
- lr: initial learning rate
- train_loss: choose a loss for training from 'adv' (sAT) or 'trades' (sTRADES)
- k: $l_0$ norm budget. In the experiments, k=120, 60, 1200, 72 for CIFAR-10, CIFAR-100, ImageNet-100 and GTSRB, respectively.
- alpha, beta: step size for updating magnitude and mask in sPGD
- n_iters: number of iterations for attack
- nfgsm: whether enabling N-FGSM
- sat: whether enabling SAT
- sat_epoch: the epoch to enable SAT, i.e., sat_epoch*max_epoch
- attack_loss: 'ce' for sTRADES (T), 'trades' for sTRADES (F)
- gpu: gpu id

## Adversarial Training
This is the implementation of adversarial training in "Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations, ICML 2024."

Run the following command to train PreActResNet18 with sAT or sTRADES:

```
python adversarial_training/train.py --exp_name debug --data_name [DATASET NAME] --data_dir [DATA PATH] --model_name preactresnet --max_epoch 100 --batch_size 128 --lr 0.05 --train_loss [adv, trades] --train_mode rand --patience 10 -k 120 --n_iters 20 --gpu 0 
```

## Sparse-AutoAttack (sAA)

Run the following command to run sAA on CIFAR10 or CIFAR100:

```
python autoattack/evaluate.py --dataset [DATASET NAME] --data_dir [DATASET PATH] --model [standard, l1, linf, l2, l0] --ckpt [CHECKPOINT NAME OR PATH] -k 20 --n_iters 10000 --n_examples 10000 --gpu 0 --bs 500 
```

- dataset: cifar10 or cifar100
- data_dir: path to the dataset (automatically download if not exist)
- model: choose a model from standard, l1, linf, l2, l0
- ckpt: checkpoint name (for robustbench models) or checkpoint path (for vanilla, $l_1$ and $l_0$ models)
- k: $l_0$ norm budget
- n_iters: number of iterations
- n_examples: number of examples to evaluate
attack


Run the following command to run sAA on ImageNet100 or GTSRB:
```
python autoattack/evaluate_large.py --dataset [DATASET NAME] --data_dir [DATASET PATH] --model [standard, l1, linf, l2, l0] --ckpt [CHECKPOINT NAME OR PATH] -k 200 --n_iters 10000 --n_examples 500 --gpu 0 --bs 64 
```
- dataset: imagenet100 or gtsrb
- data_dir: path to the dataset (please download the datasets by yourself)
- Other arguments are the same as above.

## Sparse-PGD (sPGD) / Sparse-RS (RS)
If you want to run single attack (sPGD unproj, sPGD proj or RS), run the following command:

```
python evaluate_single.py  --dataset [DATASET NAME] --data_dir [DATASET PATH]  --model [standard, l1, linf, l2, l0] --ckpt [CHECKPOINT NAME OR PATH] -k 20 --bs 500 --n_iters 10000 --n_examples 10000 --gpu 0 [--projected] [--unprojected] [--black] [--calc_aa]
```
- unprojected: run sPGD unproj
- projected: run sPGD proj
- black: run RS
- calc_aa: calculate the ensemble robust accuracy. When the all of three arguments above are set true, it is equivalent to sAA, but less efficient than evaluate.py or evaluate_large.py, because there is no cascade ensemble.

## Checkpoints
The checkpoint files of the models traiend with the proposed method are available [here](https://drive.google.com/drive/folders/1LbMRnhQ6OKy4TleHsCb9f9N15Db5aBSI?usp=sharing)


## Acknowledgement

Parts of codes are based on [DengpanFu/RobustAdversarialNetwork: A pytorch re-implementation for paper "Towards Deep Learning Models Resistant to Adversarial Attacks" (github.com)](https://github.com/DengpanFu/RobustAdversarialNetwork)

Codes of Sparse-RS are from [fra31/sparse-rs: Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks (github.com)](https://github.com/fra31/sparse-rs)

## Bibliography

If you find this repository helpful for your project, please consider citing:
```
@article{zhong2025understanding,
  title={Understanding and Improving Fast Adversarial Training against $l_0$ Bounded Perturbations},
  author={Xuyang Zhong and Yixiao Huang and Chen Liu},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

```
@inproceedings{
	zhong2024towards,
	title={Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations},
	author={Xuyang Zhong and Yixiao Huang and Chen Liu},
	booktitle={International Conference on Machine Learning},
	year={2024},
	organization={PMLR}
}
```