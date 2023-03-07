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


def benchmark(
    model: nn.Module = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10),
    img_size: int = 224,
    mode: str = "train",
    batch_size: int = 32,
    num_epochs: int = 20,
    num_workers: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    device: int = 0,
) -> None:
    assert isinstance(model, nn.Module)
    assert isinstance(img_size, int) and img_size > 0
    assert mode in ["train", "eval"]
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(num_epochs, int) and num_epochs > 0
    assert isinstance(learning_rate, float) and learning_rate > 0.0

    download = not os.path.exists(os.path.join(".", "imagenette2-320"))
    if download:
        print("Data not found. Now download it.")
    dataset = ImageNette(split=mode, size=img_size, download=download)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=mode == "train",
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, verbose=True)

    device = torch.device(device)
    model = model.to(device)

    if mode == "train":
        _train(model, dataloader, num_epochs, optimizer, scheduler, device)
    else:
        _eval(model, dataloader, num_epochs)


def _train(model, dataloader, num_epochs, optimizer, scheduler, device):
    model.train()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_fn = loss_fn.to(device)

    for i in range(num_epochs):
        costs, times = [], []
        cost_, time_ = 0.0, 0.0
        for (imgs, labels) in tqdm(dataloader):
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
            cost_ += loss.detach().item()

        scheduler.step()
        print(f"epoch: {i}; cost: {cost_:.2f}; time: {time_:.2f}")
        costs.append(cost_)
        times.append(time_)

    print(f"mean cost: {np.round(np.mean(costs), decimals=2)}; mean time: {np.round(np.mean(times), decimals=2)}")


def _eval(model, dataloader, num_epochs, device):
    model.eval()

    for i in range(num_epochs):
        scores, times = [], []
        y_preds, y_trues, time_ = [], [], 0.0
        for (imgs, labels) in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

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

        print(f"epoch: {i}; acc: {score:.2f}; time: {time_:.2f}")
        scores.append(score)
        times.append(time_)

    print(f"mean acc: {np.round(np.mean(scores), decimals=2)}; mean time: {np.round(np.mean(times), decimals=2)}")


if __name__ == "__main__":
    benchmark()
