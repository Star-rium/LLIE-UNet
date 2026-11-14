from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re

class LOLv2Dataset(Dataset):
    def __init__(
        self,
        root,
        split="Train",
        patch_size=256,
        augment=False
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment

        self.low_dir = self.root / split / "Low"
        self.normal_dir = self.root / split / "Normal"

        def get_image_map(directory):
            image_map = {}

            for path in directory.glob("*"):
                if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue

                # Extract numeric ID from filename
                match = re.search(r"\d+", path.stem)

                if match is None:
                    raise ValueError(f"Cannot extract ID: {path.name}")

                image_id = match.group()

                if image_id in image_map:
                    raise ValueError(f"Duplicate image ID: {image_id}")

                image_map[image_id] = path

            return image_map

        low_files = get_image_map(self.low_dir)
        normal_files = get_image_map(self.normal_dir)

        low_ids = set(low_files)
        normal_ids = set(normal_files)

        if low_ids != normal_ids:
            raise RuntimeError(
                f"Unmatched pairs!\n"
                f"Missing Normal: {low_ids - normal_ids}\n"
                f"Missing Low: {normal_ids - low_ids}"
            )

        self.pairs = [
            (low_files[key], normal_files[key])
            for key in sorted(low_ids, key=int)
        ]

        print(f"{split}: {len(self.pairs)} image pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def _paired_transform(self, low, normal):
        if self.augment:
            width, height = low.size

            if width < self.patch_size or height < self.patch_size:
                raise ValueError("Image smaller than patch_size")

            left = random.randint(0, width - self.patch_size)
            top = random.randint(0, height - self.patch_size)

            box = (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size
            )

            low = low.crop(box)
            normal = normal.crop(box)

            if random.random() < 0.5:
                low = low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                normal = normal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                low = low.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                normal = normal.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return low, normal

    @staticmethod
    def _to_tensor(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index):
        low_path, normal_path = self.pairs[index]

        low = Image.open(low_path).convert("RGB")
        normal = Image.open(normal_path).convert("RGB")

        if low.size != normal.size:
            raise ValueError(
                f"Image size mismatch: {low_path.name}"
            )

        low, normal = self._paired_transform(low, normal)

        return self._to_tensor(low), self._to_tensor(normal)
