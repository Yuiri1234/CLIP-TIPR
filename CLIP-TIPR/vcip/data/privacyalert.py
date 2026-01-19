import glob
import os

import pandas as pd
import torch.utils.data as data
from configs.config import CONF


class PrivacyAlertText(data.Dataset):
    def __init__(self, root, split="train", tag="utdt", type="text"):
        self.root = root
        if type == "text":
            path = os.path.join(self.root, "ImgTags", tag, f"dt_plus_ut_{split}1_2.tsv")
            df = pd.read_table(
                path, header=None, index_col=0, names=["privacy", "Filename", "tag"]
            )

            self.filename = df["Filename"].values.tolist()
            self.labels = df["privacy"].values.tolist()
            self.sentences = df["tag"].values.tolist()

        elif type == "caption":
            path = os.path.join(self.root, "annotation", f"{split}_{tag}.csv")
            df = pd.read_csv(path)
            self.filename = df["id"].values.tolist()
            self.labels = df["label"].values.tolist()
            self.sentences = df["caption"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "filename": self.filename[index],
            "target": self.labels[index],
            "text": self.sentences[index],
        }


class PrivacyAlertImage(data.Dataset):
    def __init__(self, split="train"):
        folder_path = os.path.join(CONF.PATH.DATASET, "privacy_detection_dataset_v2")

        df = pd.read_csv(
            os.path.join(
                folder_path, "Dataset_split", f"{split}_with_labels_2classes.csv"
            ),
            index_col="Filename",
        )
        names = {"Public": 0, "Private": 1}
        df["privacy"] = df["privacy"].map(names)

        self.image_paths = glob.glob(
            os.path.join(folder_path, "images", split, "*.jpg"), recursive=True
        )

        self.labels = []
        for image_path in self.image_paths:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            self.labels.append(df.loc[int(filename), "privacy"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
        }
