from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
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
        input_image, target_image = augments["image0"], augments["image"]

        input_image = transform_only_sketch(image=input_image)["image"]
        target_image = transform_only_photos(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = SketchDataset(root_dir_photos="G:/Gans/Pix2Pix/Data/CUHK_testing_photo/testing photo", root_dir_sketch="G:/Gans/Pix2Pix/Data/CUHK_testing_sketch/testing sketch")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
