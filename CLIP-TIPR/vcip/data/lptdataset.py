import ast
import json
import os
import re

import pandas as pd
import torch.utils.data as data
from configs.config import CONF


class LPTDatasetText(data.Dataset):
    def __init__(self, root, split="train", type="default"):
        assert split in ["train", "val", "test"]
        self.root = root  # TODO
        self.type = type

        path = os.path.join(self.root, self.type, f"{split}_data.csv")
        self.df = pd.read_csv(path)

        self.filename = self.df.index.values.tolist()
        self.labels = self.df["label"].values.tolist()
        self.sentences = self.df["text"].values.tolist()
        if "score" in self.df.columns:
            self.scores = [score / 10 for score in self.df["score"].values.tolist()]

        if "multi_score" in self.type:
            attributes_text = self.load_text(
                os.path.join(self.root, self.type, "attributes.txt")
            )
            self.attributes_desc = self.text_to_dict(attributes_text)

            self.attributes = {}
            for _, attribute in self.attributes_desc.items():
                name = attribute["criteria"]
                self.attributes[name] = [
                    score / 10 for score in self.df[name].values.tolist()
                ]
        self.prompt_types = self.df["prompt"].values.tolist()
        self.prompt_types = [1 if x == "private" else 0 for x in self.prompt_types]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if "multi_score" in self.type:
            return {
                "filename": self.filename[index],
                "target": (
                    self.labels[index]
                    if "score" not in self.df.columns
                    else self.scores[index]
                ),
                "text": self.sentences[index],
                "attributes": {
                    attribute: self.attributes[attribute][index]
                    for attribute in self.get_attributes()
                },
                "prompt_type": self.prompt_types[index],
            }
        else:
            return {
                "filename": self.filename[index],
                "target": (
                    self.labels[index]
                    if "score" not in self.df.columns
                    else self.scores[index]
                ),
                "text": self.sentences[index],
                "prompt_type": self.prompt_types[index],
            }

    def get_attributes(self):
        return sorted(
            [attribute["criteria"] for attribute in self.attributes_desc.values()]
        )

    def load_text(self, file):
        with open(file) as f:
            message = f.read()
        return message

    def text_to_dict(self, text):
        text = re.sub(r"^[^{]*", "", text)
        text = re.sub(r"}[^}]*$", "}", text)
        try:
            text_dict = eval(text)
            return text_dict
        except Exception:
            try:
                text_dict = ast.literal_eval(text)
                return text_dict
            except Exception:
                text_dict = json.loads(text)
                return text_dict


class LPTDatasetImage(data.Dataset):
    def __init__(self, type="default"):
        folder_path = os.path.join(CONF.PATH.LPTDATASET, "image_generation")

        df = pd.read_csv(os.path.join(folder_path, type, "image_data.csv"))

        self.image_paths = (
            df["filenumber"]
            .apply(lambda x: os.path.join(folder_path, type, f"{x}.png"))
            .values.tolist()
        )
        self.labels = df["label"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
        }
