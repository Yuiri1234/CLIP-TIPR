import glob
import os

import torch.utils.data as data
from configs.config import CONF


class COCO(data.Dataset):
    def __init__(self):
        folder_path = os.path.join(CONF.PATH.DATASET, "coco")
        self.image_paths = glob.glob(folder_path + "/**/*.*", recursive=True)
        self.labels = [0 for _ in range(len(self.image_paths))]  # すべて0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
        }
