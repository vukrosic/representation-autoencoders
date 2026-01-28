import torch
import torchvision.transforms as T
from PIL import Image
import os

def create_dummy_dataset(path="./data/train/dummy", num_images=50):
    os.makedirs(path, exist_ok=True)
    for i in range(num_images):
        img = torch.randn(3, 224, 224)
        img = T.ToPILImage()(img)
        img.save(os.path.join(path, f"img_{i}.png"))
    print(f"Created {num_images} dummy images at {path}")

if __name__ == "__main__":
    create_dummy_dataset()
