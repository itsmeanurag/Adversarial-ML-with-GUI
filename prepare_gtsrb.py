import os
import numpy as np
from PIL import Image

IMG_SIZE = (32, 32)
data_dir = "datasets/gtsrb/Final_Training/Images"

images, labels = [], []
for label in sorted(os.listdir(data_dir), key=lambda x: int(x)):
    class_folder = os.path.join(data_dir, label)
    if not os.path.isdir(class_folder):
        continue
    for fname in os.listdir(class_folder):
        if fname.endswith('.ppm'):
            img_path = os.path.join(class_folder, fname)
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(arr)
            labels.append(int(label))

images = np.stack(images)
labels = np.array(labels)
np.savez_compressed("datasets/gtsrb/train.npz", images=images, labels=labels)
print("Saved datasets/gtsrb/train.npz")