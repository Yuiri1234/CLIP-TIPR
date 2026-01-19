import os

import pandas as pd
import torch.utils.data as data


class ETHICS(data.Dataset):
    def __init__(self, root, split="train", filtering=None):
        assert split in ["train", "test", "test_hard", "ambig"]
        self.root = root  # ethics/commonsense

        path = os.path.join(self.root, f"cm_{split}.csv")
        df = pd.read_csv(path)

        if filtering == "short":
            df = df[df["is_short"]]

        if filtering == "long":
            df = df[not df["is_short"]]

        if split == "ambig":
            self.labels = [-1 for _ in range(df.shape[0])]
            self.sentences = [df.iloc[i, 0] for i in range(df.shape[0])]
        else:
            self.labels = [df.iloc[i, 0] for i in range(df.shape[0])]
            self.sentences = [df.iloc[i, 1] for i in range(df.shape[0])]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "filename": index,
            "target": self.labels[index],
            "text": self.sentences[index],
        }
