import torch
from torch import optim, nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from sklearn import metrics

from dataset import ImageNette
import timm

import os
from time import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Benchmark timm models on ImageNette with DDP.")
parser.add_argument(
    "--model",
    type=str,
    default="vit_base_patch16_224"
)
parser.add_argument(
    "--image_size",
    type=int,
    default=224,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=25
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=5
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1
)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def benchmark(
    local_rank: int,
    nprocs: int,
    args: argparse.ArgumentParser
) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    setup(local_rank, nprocs)
    print(f"Initialized successfully.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"

    model = timm.create_model(args.model, pretrained=True, num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=1e-6, verbose=True)

    num_workers = args.num_workers // nprocs
    batch_size = args.batch_size // nprocs

    train_dataset = ImageNette(split="train", size=args.image_size)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )
    if local_rank == 0:
        model_without_ddp = model
        val_dataset = ImageNette(split="val", size=args.image_size)
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0
        )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    costs, scores, times = [], [], []
    for i in range(args.num_epochs):
        cost_, y_preds, y_trues, time_ = [], [], [], 0.0

        model.train()
        if local_rank == 0:
            print(f"Epoch: {i}")
            print("Training")
        for (imgs, labels) in tqdm(train_dataloader):
            imgs, labels = imgs.to(device), labels.long().to(device)

            tic = time()
            with torch.enable_grad():
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                loss = reduce_mean(loss, nprocs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            toc = time()
            time_ += toc - tic
            cost_.append(loss.detach().item())

        scheduler.step()
        torch.cuda.synchronize()

        if local_rank == 0:
            cost_ = np.mean(cost_)
            if i >= args.warmup_epochs:
                costs.append(cost_)

            model_without_ddp.load_state_dict(model.module.state_dict())
            model_without_ddp.eval()
            print("Evaluating")
            for (imgs, labels) in tqdm(val_dataloader):
                imgs = imgs.to(device)

                tic = time()
                with torch.no_grad():
                    preds = torch.softmax(model(imgs), dim=1)
                toc = time()
                time_ += toc - tic

                y_preds.append(preds.cpu().numpy())
                y_trues.append(labels.numpy())

            y_preds = np.concatenate(y_preds, axis=0)
            y_trues = np.concatenate(y_trues, axis=0)
            y_preds = np.argmax(y_preds, axis=1)
            score = metrics.accuracy_score(y_pred=y_preds, y_true=y_trues)

            print(f"cost: {cost_:.3f}; acc: {score:.3f}; time: {time_:.3f}")
            if i >= args.warmup_epochs:
                scores.append(score)
                times.append(time_)

        dist.barrier()

    if local_rank == 0:
        print("Benchmarking finished.")
        print(f"Lowest cost: {np.round(min(costs), decimals=3)}")
        print(f"Highest score: {np.round(max(scores), decimals=3)}")
        print(f"Mean time: {np.round(np.mean(times), decimals=3)}")


def setup(local_rank, nprocs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=local_rank, world_size=nprocs)


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    print(f"Number of GPUs: {args.nprocs}")

    mp.spawn(benchmark, nprocs=args.nprocs, args=(args.nprocs, args))


if __name__ == "__main__":
    main()
