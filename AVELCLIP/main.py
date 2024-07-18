#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import math
import builtins
import shutil
import time
import warnings
import clip.model
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import random
from tqdm import tqdm
import h5py
import librosa

# ----自定义dataset-------#
import torchvision.datasets

class My_DataSet(torchvision.datasets.VisionDataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.visual_path = '/home/hexiang/Encoders/data/image_opencv_VGG19_finetune/'
        self.audio_path = "/home/hexiang/CACE_lxx/data/audio_feature.h5"
        self.length = len(self.file_list)

    def __getitem__(self, mask):
        """
        return: visual_data [10, 3, 224, 224]
                audio_data [1, 321600]
        """
        file_name = self.file_list[mask]
        file_path = os.path.join(self.visual_path, file_name + '.h5')
        with h5py.File(file_path, 'r') as h5_file:
            # 假设你想从每个文件中读取名为 'data' 的数据集
            # 你需要根据你的文件结构进行调整
            data = h5_file['extract_frame'][:]
            # indices = torch.arange(0, 160, 16) + torch.randint(0, 16, (10,))
            # 将数据转换为 PyTorch 张量
            visual_data = data

        # file_path = os.path.join(self.audio_path, file_name + '.wav')
        # (audio_data, _) = librosa.core.load(file_path, sr=32000, mono=True)
        # audio_data = audio_data[:320000]
        print("------####--------:{}".format(file_name))
        print("------####--------:{}".format(type(file_name)))
        num_line = self.find_line_number(file_name)
        audio_path = os.path.join(self.audio_path)
        audio_feature = h5py.File(audio_path, 'r')['avadataset']
        audio_data = audio_feature[num_line]

        return visual_data, audio_data

    def __len__(self):
        return self.length
    
    def find_line_number(target_id):
        with open('/home/hexiang/Encoders/data/AVE_Dataset/Annotations.txt', 'r') as file:
            for i, line in enumerate(file):
                if line.split('&')[0] == target_id:
                    return i
        return None

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[15, 30],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--dim-I", default=512, type=int, help="feature dimension of image (default: 512)"
)
parser.add_argument(
    "--dim-A", default=128, type=int, help="feature dimension of audio (default: 128)"
)
parser.add_argument(
    "--hidden-dim", default=128, type=int, help="feature dimension hidden layer (default: 128)"
)
parser.add_argument(
    "--t", default=0.2, type=float, help="softmax temperature (default: 0.07)"
)
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--mel_bins', type=int, default=64)
parser.add_argument('--fmin', type=int, default=50)
parser.add_argument('--fmax', type=int, default=14000) 
parser.add_argument('--model_type', type=str, default="Cnn14")
parser.add_argument('--checkpoint_path', type=str, default="/home/hexiang/AVELCLIP/audioset_tagging_cnn/checkpoints/Cnn14_mAP=0.431.pth")
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    # ngpus_per_node = torch.cuda.device_count()
    # main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    model =clip.model.CLIP(
        args.dim_I,
        args.dim_A,
        args.hidden_dim,
        args.t,
        args.batch_size,
        args,
    )
    print(model)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    cudnn.benchmark = True

    lis = []
    # 打开并读取文件
    with open('/home/hexiang/Encoders/data/AVE_Dataset/trainSet.txt', 'r') as file:
        for line in file:
            # 分割每一行，并获取文件名（即第二部分）
            parts = line.strip().split('&')
            if len(parts) > 1:  # 确保有足够的部分来提取文件名
                file_name = parts[1]
                lis.append(file_name)
    with open('/home/hexiang/Encoders/data/AVE_Dataset/valSet.txt', 'r') as file:
        for line in file:
            # 分割每一行，并获取文件名（即第二部分）
            parts = line.strip().split('&')
            if len(parts) > 1:  # 确保有足够的部分来提取文件名
                file_name = parts[1]
                lis.append(file_name)

    train_dataset = My_DataSet(file_list=lis)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "I_state_dict": model.module.encoder_I.state_dict(),
                },
                is_best=False,
                filename="checkpoint_{:04d}.pth.tar".format(epoch),
            )

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, audio) in enumerate(train_loader):
        """
        images: [b ,t, 3, 224, 224]
        audio:[b, 1, 320000]
        """
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            audio = audio.cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(images, audio, args)
        loss1 = criterion(output, target)
        loss2 = criterion(output.T, target)

        loss = (loss1 + loss2) / 2.
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(args, output, target, topk=(1, 5))
        losses.update(loss.item(), args.batch_size)
        top1.update(acc1[0], args.batch_size)
        top5.update(acc5[0], args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(args, output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = args.batch_size

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()