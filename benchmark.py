import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

from dataset import ImageNette
import timm

from time import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Benchmark timm models on ImageNette.")
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
    warmup_epochs: int = 5,
    device: int = 0,
) -> None:
    assert isinstance(model, nn.Module)
    assert isinstance(image_size, int) and image_size > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(num_epochs, int) and num_epochs > 0
    assert isinstance(learning_rate, float) and learning_rate > 0.0

    train_dataset = ImageNette(split="train", size=image_size)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )
    val_dataset = ImageNette(split="val", size=image_size)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0
    )

    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=1e-6, verbose=True)

    device = torch.device(device)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    costs, scores, times = [], [], []
    for i in range(num_epochs):
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
        scheduler.step()

        if i >= warmup_epochs:
            costs.append(cost_)

        model.eval()
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
        if i >= warmup_epochs:
            scores.append(score)
            times.append(time_)

    print("Benchmarking finished.")
    print(f"Lowest cost: {np.round(min(costs), decimals=3)}")
    print(f"Highest score: {np.round(max(scores), decimals=3)}")
    print(f"Mean time: {np.round(np.mean(times), decimals=3)}")


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
