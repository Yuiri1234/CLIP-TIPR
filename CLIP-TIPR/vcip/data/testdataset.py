import os
import sys

import pandas as pd
from torch.utils import data

sys.path.append(".")
from configs.config import CONF  # noqa: E402


class TestDatasetImage(data.Dataset):
    def __init__(self):
        df = pd.read_csv(
            os.path.join(CONF.PATH.DATASET, "testdataset", "test_data.csv")
        )
        self.ids = df["id"].values.tolist()
        self.labels = df["label"].values.tolist()
        self.image_paths = df["path"].values.tolist()
        self.contents = df["content"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "filename": self.image_paths[index],
            "content": self.contents[index],
        }


class TestDatasetText(data.Dataset):
    def __init__(self, df):
        self.labels = df["label"].values.tolist()
        self.texts = df["text"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "text": self.texts[index],
        }
