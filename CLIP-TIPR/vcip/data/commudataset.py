import os
import sys

import pandas as pd
from torch.utils import data

sys.path.append(".")
from configs.config import CONF  # noqa: E402


class CommuDatasetImage(data.Dataset):
    def __init__(self, is_filtered=False, is_balanced=False):
        if not is_filtered:
            if not is_balanced:
                df = pd.read_csv(
                    os.path.join(CONF.PATH.DATASET, "commudataset", "all_data.csv")
                )
            else:
                df = pd.read_csv(
                    os.path.join(
                        CONF.PATH.DATASET, "commudataset", "balanced_all_data.csv"
                    )
                )
        else:
            if not is_balanced:
                df = pd.read_csv(
                    os.path.join(CONF.PATH.DATASET, "commudataset", "filtered_data.csv")
                )
            else:
                df = pd.read_csv(
                    os.path.join(
                        CONF.PATH.DATASET, "commudataset", "balanced_filtered_data.csv"
                    )
                )
        df["path"] = df["path"].apply(lambda x: os.path.join(CONF.PATH.DATASET, x))
        self.labels = df["label"].values.tolist()
        self.image_paths = df["path"].values.tolist()
        self.scenarios = df["scenario"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
            "scenario": self.scenarios[index],
        }


class CPIDataset(data.Dataset):
    def __init__(self):
        df = pd.read_csv(
            os.path.join(CONF.PATH.DATASET, "cpidataset", "sampled_data_anonymized.csv")
        )
        df["path"] = df["path"].apply(lambda x: os.path.join(CONF.PATH.DATASET, x))
        self.labels = df["label"].values.tolist()
        self.image_paths = df["path"].values.tolist()
        self.scenarios = df["scenario"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
            "scenario": self.scenarios[index],
        }


class CommuDatasetText(data.Dataset):
    def __init__(self, df=None):
        if df is None:
            df = pd.read_csv(
                os.path.join(CONF.PATH.DATASET, "commudataset", "different_phrase.csv")
            )
        try:
            self.senarios = df["scenario"].values.tolist()
            self.types = df["type"].values.tolist()
        except KeyError:
            self.senarios = None
            self.types = None
        try:
            if "score" in df.columns:
                self.labels = df["score"].values.tolist()
            elif "label" in df.columns:
                self.labels = df["label"].values.tolist()
        except KeyError:
            self.labels = None
        self.texts = df["text"].values.tolist()
        if "meaning" in df.columns:
            self.meanings = df["meaning"].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if "meaning" in self.__dict__:
            return {
                "scenario": self.senarios[index],
                "type": self.types[index],
                "text": self.texts[index],
                "meaning": self.meanings[index],
            }
        elif self.senarios is not None:
            return {
                "scenario": self.senarios[index],
                "type": self.types[index],
                "text": self.texts[index],
            }
        if self.labels is not None:
            return {
                "target": self.labels[index],
                "text": self.texts[index],
            }
        else:
            return {
                "text": self.texts[index],
            }
