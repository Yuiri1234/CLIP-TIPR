import glob
import os

import torch.utils.data as data
from configs.config import CONF


class VCI(data.Dataset):
    def __init__(self):
        folder_path = os.path.join(CONF.PATH.DATASET, "VCI")
        self.image_paths = glob.glob(folder_path + "/**/*.*", recursive=True)
        self.labels = [1 for _ in range(len(self.image_paths))]  # すべて1

        self.contents = [path.split(os.path.sep)[-3] for path in self.image_paths]
        self.details = [path.split(os.path.sep)[-2] for path in self.image_paths]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
            # "content": self.contents[index],
            # "detail": self.details[index],
        }
