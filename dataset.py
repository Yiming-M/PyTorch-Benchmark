import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomResizedCrop, Resize, Normalize, Compose

from PIL import Image

import os
from glob import glob
import requests
import tarfile

from typing import Tuple

url = {
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
}


class ImageNette(Dataset):
    def __init__(self, split: bool, size: int, download: bool = True) -> None:
        super().__init__()
        assert split in ["train", "val"]
        assert isinstance(size, int) and size > 0
        if download:
            with requests.get(url["imagenette"], stream=True) as tgz_file, tarfile.open(fileobj=tgz_file.raw, mode="r:gz") as tar_obj:
                tar_obj.extractall()

        self.split = split
        self.size = size

        self.data_dir_path = os.path.join(".", "imagenette2-320", self.split)
        self.__make_data__()

        resize_transform = RandomResizedCrop if self.split == "train" else Resize
        self.transform = Compose([
            ToTensor(),
            resize_transform(size=(self.size, self.size),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

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
            img_ = img.convert("RGB")
            x = self.transform(img_)

        y = torch.tensor(label)

        return x, y
