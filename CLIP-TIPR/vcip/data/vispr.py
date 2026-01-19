import ast
import csv
import glob
import json
import os
import sys
from collections import Counter

import pandas as pd
import torch.utils.data as data

sys.path.append(".")
from configs.config import CONF  # noqa: E402

privacy_attributes = {
    "full nudity": "Nudity",
    "credit card": "Credit Card",
    "address home complete": "Home address (C)",
    "passport": "Passport",
    "ausweis": "National Identification",
    "fingerprint": "Fingerprint",
    "semi nudity": "Semi-nudity",
    "signature": "Signature",
    "phone": "Phone no.",
    "drivers license": "Drivers License",
    "license plate complete": "License Plate (C)",
    "rel personal": "Relationships",
    "legal involvement": "Legal involvement",
    "email content": "Email content",
    "address home partial": "Home address (P)",
    "injury": "Medical History",
    "face partial": "Face (P)",
    "face complete": "Face (C)",
    "mail": "Mail",
    "student id": "Student ID",
    "email": "Email address",
    "rel professional": "Professional Circle",
    "address current complete": "Visited Location (C)",
    "license plate partial": "License Plate (P)",
    "receipt": "Receipts",
    "birth date": "Date of Birth",
    "address current partial": "Visited Location (P)",
    "name full": "Name (Full)",
    "username": "Username",
    "ticket": "Tickets",
    "medicine": "Medical Treatment",
    "online conversation": "Online conversations",
    "date time": "Date/Time of Activity",
    "disability physical": "Physical disability",
    "rel social": "Social Circle",
    "name last": "Name (Last)",
    "occassion personal": "Occassion",
    "education history": "Education history",
    "vehicle ownership": "Vehicle Ownership",
    "handwriting": "Handwriting",
    "opinion political": "Political Opinion",
    "opinion general": "General Opinion",
    "occassion work": "Work Occassion",
    "birth city": "Place of Birth",
    "rel views": "Similar view",
    "landmark": "Landmark",
    "tattoo": "Tattoo",
    "weight approx": "Weight Group",
    "rel competitors": "Competitors",
    "rel spectators": "Spectators",
    "name first": "Name (First)",
    "sexual orientation": "Sexual Orientation",
    "occupation": "Occupation",
    "religion": "Religion",
    "race": "Race",
    "color": "Color",
    "nationality": "Nationality",
    "ethnic clothing": "Traditional clothing",
    "sports": "Sports",
    "height approx": "Height Group",
    "culture": "Culture",
    "marital status": "Marital status",
    "hobbies": "Hobbies",
    "gender": "Gender",
    "eye color": "Eye color",
    "age approx": "Age Group",
    "hair color": "Hair color",
    "safe": "Safe",
}


class VISPR(data.Dataset):
    def __init__(self, split="train", filename=None, threshold=1, multi_label=False):
        folder_path = os.path.join(CONF.PATH.DATASET, "VISPR")

        if not multi_label:
            # df = pd.read_csv(
            #     os.path.join(folder_path, "annos", f"{split}2017.csv"),
            #     index_col="image_path",
            # )
            # if filename is None:
            #     self.image_paths = glob.glob(
            #         os.path.join(folder_path, "images", f"{split}2017", "*.jpg"),
            #         recursive=True,
            #     )

            #     self.labels = []
            #     for image_path in self.image_paths:
            #         filename = os.path.basename(image_path)
            #         self.labels.append(df.loc[filename, "label"])
            # else:
            #     self.image_paths = [
            #         os.path.join(folder_path, "images", f"{split}2017", filename)
            #     ]
            #     self.labels = [df.loc[filename, "label"]]
            df = pd.read_csv(
                os.path.join(folder_path, "annos", "multilabel", f"{split}2017.csv"),
                index_col="image_path",
            )
            if filename is None:
                self.image_paths = glob.glob(
                    os.path.join(folder_path, "images", f"{split}2017", "*.jpg"),
                    recursive=True,
                )

                self.labels = []
                for image_path in self.image_paths:
                    filename = os.path.basename(image_path).split(".")[0]
                    labels = df.loc[filename, "label"]
                    if isinstance(labels, str):
                        labels = ast.literal_eval(labels)
                    self.labels.append(
                        1
                        if any(1 if label > threshold else 0 for label in labels)
                        else 0
                    )
            else:
                self.image_paths = [
                    os.path.join(folder_path, "images", f"{split}2017", filename)
                ]
                labels = df.loc[filename, "label"]
                if isinstance(labels, str):
                    labels = ast.literal_eval(labels)
                self.labels = [
                    1 if any(1 if label > threshold else 0 for label in labels) else 0
                ]
        else:
            df = pd.read_csv(
                os.path.join(folder_path, "annos", "multilabel", f"{split}2017.csv"),
                index_col="image_path",
            )
            if filename is None:
                self.image_paths = glob.glob(
                    os.path.join(folder_path, "images", f"{split}2017", "*.jpg"),
                    recursive=True,
                )

                self.labels = []
                self.captions = []
                for image_path in self.image_paths:
                    filename = os.path.basename(image_path).split(".")[0]
                    label = df.loc[filename, "label"]
                    self.labels.append(label)
            else:
                self.image_paths = [
                    os.path.join(folder_path, "images", f"{split}2017", filename)
                ]
                self.labels = [df.loc[filename, "label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "image_path": self.image_paths[index],
        }


class VISPRText(data.Dataset):
    def __init__(self, split="train", filename=None, threshold=1, multi_label=False):
        folder_path = os.path.join(CONF.PATH.DATASET, "VISPR")
        path = os.path.join(folder_path, "annotation", f"{split}.csv")
        caption_df = pd.read_csv(path, index_col="id")

        if not multi_label:
            df = pd.read_csv(
                os.path.join(folder_path, "annos", "multilabel", f"{split}2017.csv"),
                index_col="image_path",
            )
            if filename is None:
                self.image_paths = glob.glob(
                    os.path.join(folder_path, "images", f"{split}2017", "*.jpg"),
                    recursive=True,
                )

                self.labels = []
                self.captions = []
                for image_path in self.image_paths:
                    filename = os.path.basename(image_path).split(".")[0]
                    labels = df.loc[filename, "label"]
                    if isinstance(labels, str):
                        labels = ast.literal_eval(labels)
                    self.labels.append(
                        1
                        if any(1 if label > threshold else 0 for label in labels)
                        else 0
                    )
                    self.captions.append(caption_df.loc[filename, "caption"])
            else:
                self.image_paths = [
                    os.path.join(folder_path, "images", f"{split}2017", filename)
                ]
                labels = df.loc[filename, "label"]
                if isinstance(labels, str):
                    labels = ast.literal_eval(labels)
                self.labels.append(
                    1 if any(1 if label > threshold else 0 for label in labels) else 0
                )
        else:
            df = pd.read_csv(
                os.path.join(folder_path, "annos", "multilabel", f"{split}2017.csv"),
                index_col="image_path",
            )
            if filename is None:
                self.image_paths = glob.glob(
                    os.path.join(folder_path, "images", f"{split}2017", "*.jpg"),
                    recursive=True,
                )

                self.labels = []
                for image_path in self.image_paths:
                    filename = os.path.basename(image_path).split(".")[0]
                    label = df.loc[filename, "label"]
                    self.labels.append(label)
            else:
                self.image_paths = [
                    os.path.join(folder_path, "images", f"{split}2017", filename)
                ]
                self.labels = [df.loc[filename, "label"]]
                self.captions = [caption_df.loc[filename, "caption"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "target": self.labels[index],
            "filename": self.image_paths[index],
            "text": self.captions[index],
        }


def create_csvfile(input_dir, output_csv, multi_label=False):
    csv_header = ["image_path", "label"]

    csv_data = []

    if not multi_label:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, "r") as json_file:
                        data = json.load(json_file)
                        # safe or not safe
                        if "labels" in data:
                            if "a0_safe" in data["labels"]:
                                label = 0
                            else:
                                label = 1
                        else:
                            label = 0
                        csv_data.append(
                            [data["image_path"].split("/")[-1].split(".")[0], label]
                        )
    else:
        privacy_distribution_path = os.path.join(
            CONF.PATH.VISPR, "train2017_privacy_distribution.csv"
        )
        if not os.path.exists(privacy_distribution_path):
            anno_dirs = glob.glob(
                os.path.join(CONF.PATH.DATASET, "VISPR", "images", "*_anno")
            )
            for dir in anno_dirs:
                files = glob.glob(os.path.join(dir, "*"))
                data_type = dir.split("/")[-1].split("_")[0]

                all_dict = []
                for file in files:
                    with open(file, "r") as f:
                        data = json.load(f)
                    all_dict.append(data)
                all_df = pd.DataFrame(all_dict)

                label_list = []
                for i in range(len(all_df)):
                    label_list.extend(all_df["labels"][i])
                c = Counter(label_list)

                c_df = pd.DataFrame(c.most_common(), columns=["label", "count"])
                c_df["index"] = c_df["label"].apply(lambda x: int(x.split("_")[0][1:]))
                c_df["name"] = c_df["label"].apply(lambda x: " ".join(x.split("_")[1:]))
                c_df["fixed_name"] = c_df["name"].apply(lambda x: privacy_attributes[x])
                c_df["privacy_score"] = c_df["name"].apply(
                    lambda x: len(privacy_attributes)
                    - list(privacy_attributes.keys()).index(x)
                )
                c_df = c_df.sort_values("privacy_score", ascending=False)
                c_df = c_df.reset_index(drop=True)
                c_df.to_csv(
                    os.path.join(
                        CONF.PATH.VISPR, f"{data_type}_privacy_distribution.csv"
                    ),
                    index=False,
                )

        label_df = pd.read_csv(
            os.path.join(CONF.PATH.VISPR, "train2017_privacy_distribution.csv"),
            index_col="label",
        )
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, "r") as json_file:
                        data = json.load(json_file)
                        # multi label
                        if "labels" in data:
                            labels = [
                                label_df.loc[label, "privacy_score"].item()
                                for label in data["labels"]
                            ]
                        else:
                            labels = [1]
                        csv_data.append(
                            [data["image_path"].split("/")[-1].split(".")[0], labels]
                        )

    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(csv_data)

    print(f"CSVファイル {output_csv} が作成されました。")


if __name__ == "__main__":
    folder_path = os.path.join(CONF.PATH.DATASET, "VISPR")
    input_paths = [
        os.path.join(folder_path, f"{split}2017") for split in ["train", "val", "test"]
    ]
    output_paths1 = [
        os.path.join(folder_path, "annos", "normal", f"{split}2017.csv")
        for split in ["train", "val", "test"]
    ]
    output_paths2 = [
        os.path.join(folder_path, "annos", "multilabel", f"{split}2017.csv")
        for split in ["train", "val", "test"]
    ]
    for input_path, output_path in zip(input_paths, output_paths1):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        create_csvfile(input_path, output_path, multi_label=False)
    for input_path, output_path in zip(input_paths, output_paths2):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        create_csvfile(input_path, output_path, multi_label=True)
