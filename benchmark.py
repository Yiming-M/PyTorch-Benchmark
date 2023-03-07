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
    model: nn.Module = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10),
    img_size: int = 224,
    batch_size: int = 32,
    num_epochs: int = 20,
    num_workers: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    device: int = 0,
) -> None:
    assert isinstance(model, nn.Module)
    assert isinstance(img_size, int) and img_size > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(num_epochs, int) and num_epochs > 0
    assert isinstance(learning_rate, float) and learning_rate > 0.0

    download = not os.path.exists(os.path.join(".", "imagenette2-320"))
    if download:
        print("Data not found. Now download it.")
    train_dataset = ImageNette(split="train", size=img_size, download=download)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )
    val_dataset = ImageNette(split="val", size=img_size, download=download)
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

        scheduler.step()
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

        print(f"cost: {cost_:.2f}; acc: {score:.2f}; time: {time_:.2f}")
        scores.append(score)
        times.append(time_)

    print("Benchmarking finished.")
    print(f"Costs: {costs}")
    print(f"Scores: {scores}")
    print(f"Mean time: {np.round(np.mean(times), decimals=2)}")


if __name__ == "__main__":
    benchmark()
