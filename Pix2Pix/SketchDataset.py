from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0": "image"}
)

transform_only_sketch = A.Compose(
    [
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

transform_only_photos = A.Compose(
    [
        A.ColorJitter(p=0.11),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


class SketchDataset(Dataset):
    def __init__(self, root_dir_photos, root_dir_sketch):
        self.root_dir_photos = root_dir_photos
        self.root_dir_sketch = root_dir_sketch
        self.list_photos = os.listdir(self.root_dir_photos)
        self.list_sketch = os.listdir(self.root_dir_sketch)

    def __len__(self):
        assert len(self.list_photos) == len(self.list_sketch)
        return len(self.list_photos)

    def __getitem__(self, index):
        photo_file = self.list_photos[index]
        sketch_file = self.list_sketch[index]
        photo_path = os.path.join(self.root_dir_photos, photo_file)
        sketch_path = os.path.join(self.root_dir_sketch, sketch_file)
        photo = np.array(Image.open(photo_path))
        sketch = np.array(Image.open(sketch_path))

        augments = both_transform(image=photo, image0=sketch)
        input_image, target_image = augments["image"], augments["image0"]

        input_image = transform_only_sketch(image=input_image)["image"]
        target_image = transform_only_photos(image=target_image)["image"]

        return input_image, target_image
