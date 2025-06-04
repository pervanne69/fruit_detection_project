import os
import numpy as np
import cv2
from tqdm import tqdm

def generate_noise_images(out_dir, num_images=500, size=(224, 224)):
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(num_images)):
        noise = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"noise_{i}.jpg"), noise)

if __name__ == '__main__':
    generate_noise_images("../data/train/empty", 500)
    generate_noise_images("../data/val/empty", 100)
