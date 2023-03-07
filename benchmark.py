import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

from dataset import ImageNette
import timm

import os
from time import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Benchmark timm models on ImageNette.")
parser.add_argument(
    "--model",
    type=str,
    default="vgg16_bn"
)
parser.add_argument(
    "--image-size",
    type=int,
    default=224,
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32
)
parser.add_argument(
    "--num-epochs",
    type=int,
    default=20
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=2
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--device",
    type=int,
    default=0
)


def benchmark(
    model: nn.Module = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10),
    image_size: int = 224,
    batch_size: int = 32,
    num_epochs: int = 20,
    num_workers: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    device: int = 0,
) -> None:
    assert isinstance(model, nn.Module)
    assert isinstance(image_size, int) and image_size > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(num_epochs, int) and num_epochs > 0
    assert isinstance(learning_rate, float) and learning_rate > 0.0

    download = not os.path.exists(os.path.join(".", "imagenette2-320"))
    if download:
        print("Data not found. Now download it.")
    train_dataset = ImageNette(split="train", size=image_size, download=download)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )
    val_dataset = ImageNette(split="val", size=image_size, download=download)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=1e-6, verbose=True)

    device = torch.device(device)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    for i in range(num_epochs):
        costs, scores, times = [], [], []
        cost_, y_preds, y_trues, time_ = [], [], [], 0.0

        model.train()
        print(f"Epoch: {i}")
        print("Training")
        for (imgs, labels) in tqdm(train_dataloader):
            imgs, labels = imgs.to(device), labels.long().to(device)

            tic = time()
            with torch.enable_grad():
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            toc = time()

            time_ += toc - tic
            cost_.append(loss.detach().item())

        cost_ = np.mean(cost_)
        costs.append(cost_)

        model.eval()
        print("Evaluating")
        for (imgs, labels) in tqdm(val_dataloader):
            imgs = imgs.to(device)

            tic = time()
            with torch.no_grad():
                preds = torch.softmax(model(imgs), dim=1)
            toc = time()

            y_preds.append(preds.cpu().numpy())
            y_trues.append(labels.numpy())

            time_ += toc - tic

        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.argmax(y_preds, axis=1)
        score = metrics.accuracy_score(y_pred=y_preds, y_true=y_trues)

        print(f"cost: {cost_:.3f}; acc: {score:.3f}; time: {time_:.3f}")
        scores.append(score)
        times.append(time_)

        scheduler.step()

    print("Benchmarking finished.")
    print(f"Costs: {costs}")
    print(f"Scores: {scores}")
    print(f"Mean time: {np.round(np.mean(times), decimals=2)}")


if __name__ == "__main__":
    args = parser.parse_args()
    args.model = timm.create_model(args.model, pretrained=True, num_classes=10)
    benchmark(
        model=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )
