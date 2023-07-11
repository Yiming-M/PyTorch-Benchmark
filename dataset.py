import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomResizedCrop, Resize, Normalize, Compose

from PIL import Image

import os
from glob import glob
import requests
from tqdm import tqdm
import tarfile

from typing import Tuple

url = {
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
}


class ImageNette(Dataset):
    def __init__(self, split: bool, size: int) -> None:
        super().__init__()
        assert split in ["train", "val"]
        assert isinstance(size, int) and size > 0
        self.split = split
        self.size = size

        self.data_dir_path = os.path.join(".", "imagenette2-320", self.split)
        if not os.path.exists(self.data_dir_path):
            self.__download__()
        self.__make_data__()

        resize_transform = RandomResizedCrop if self.split == "train" else Resize
        self.transform = Compose([
            ToTensor(),
            resize_transform(size=(self.size, self.size), antialias=True),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __download__(self) -> None:
        print("Data not found. Now download it.")
        resp = requests.get(url["imagenette"], stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open("imagenette2-320.tgz", "wb") as file, tqdm(
            desc="Downloading imagenette2-320.tgz",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print("Downloading finished.")

        file = tarfile.open("imagenette2-320.tgz")
        file.extractall()
        file.close()
        print("Data extracted.")
        os.remove("imagenette2-320.tgz")

    def __make_data__(self) -> None:
        img_paths, labels = [], []
        categories = os.listdir(self.data_dir_path)
        categories.sort()
        for i in range(len(categories)):
            img_paths_ = glob(os.path.join(self.data_dir_path, categories[i], "*.JPEG"))
            img_paths += img_paths_
            labels_ = [i] * len(img_paths_)
            labels += labels_

        assert len(img_paths) == len(labels)
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
        x = self.transform(img)

        y = torch.tensor(label)

        return x, y
