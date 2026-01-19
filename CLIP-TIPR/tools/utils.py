import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score


def fix_seed(seed=None):
    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"

    # reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    g = None

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_vote(result_df, k=5, only_num_of_ones=False):
    """
    Calculate the predicted labels based on the k-nearest neighbors voting mechanism.
    Parameters:
    result_df (pd.DataFrame): DataFrame containing the true labels and the high
                              similarity labels.
    k (int): Number of nearest neighbors to consider for voting. Default is 5.
    only_num_of_ones (bool): If True, only return the number of ones in the predicted
                             labels for each sample. Default is False.
    Returns:
    tuple: A tuple containing:
        - y_true (np.ndarray): Array of true labels.
        - y_pred_class (np.ndarray): Array of predicted classes (0 or 1) based on
                                     the voting mechanism.
        - num_of_ones (np.ndarray): Array of the count of ones in the predicted
                                    labels for each sample.
    """
    y_true = result_df["image_label"].values
    y_pred = result_df[[f"high_sim_label_{i+1}" for i in range(k)]].values

    num_of_ones = np.sum(y_pred, axis=1)

    y_pred_class = np.where(num_of_ones <= k // 2, 0, 1)

    if only_num_of_ones:
        return num_of_ones

    return y_true, y_pred_class, num_of_ones


def create_confusion_matrix(
    y_true,
    y_pred,
    store_image_path,
    vote_each=False,
    num_of_ones=None,
    k=None,
    return_metrics=True,
    knn_regressor=False,
    fmeasure_alpha=False,
):
    """
    Create and save a confusion matrix as an image file, and optionally
    return accuracy and F1-score metrics.
    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    store_image_path (str): Path to save the confusion matrix image.
    vote_each (bool, optional): If True, create a confusion matrix for
                                the number of '1's in predictions.
                                Default is False.
    num_of_ones (list or array-like, optional): Number of '1's in each prediction.
                                                Required if vote_each is True.
    k (int, optional): The maximum number of '1's in predictions. Required
                       if vote_each is True.
    return_metrics (bool, optional): If True, return accuracy and F1-score metrics.
                                     Default is True.
    Returns:
    tuple: (accuracy, fmeasure) if return_metrics is True.
    """
    fig = plt.figure(figsize=(8, 6))
    cm_labels = ["Public", "Private"]
    if knn_regressor:
        cm = np.zeros((2, 11), dtype=int)

        for i in range(len(y_true)):
            cm[int(y_true[i]), int(y_pred[i] * 10)] += 1

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=range(11),
            yticklabels=cm_labels,
            annot_kws={"size": 18},
        )
        plt.xlabel(f"Prediction Label (Top-{k})", fontsize=15)
        y_pred = np.array([1 if pred >= 0.3 else 0 for pred in y_pred])
    elif vote_each:
        cm = np.zeros((2, k + 1), dtype=int)

        for i in range(len(y_true)):
            cm[int(y_true[i]), int(num_of_ones[i])] += 1

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=range(k + 1),
            yticklabels=cm_labels,
            annot_kws={"size": 18},
        )
        plt.xlabel(f"Num of Private (Top-{k})", fontsize=15)
    else:
        cm = confusion_matrix(
            y_true,
            y_pred,
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            annot_kws={"size": 30},
        )
        plt.xlabel("Prediction Label", fontsize=20)

    accuracy = accuracy_score(y_true, y_pred)
    f1_score_binary = f1_score(y_true, y_pred, average="binary")
    f1_score_macro = f1_score(y_true, y_pred, average="macro")
    if fmeasure_alpha:
        fmeasure02 = fbeta_score(y_true, y_pred, average="binary", beta=0.2)

    plt.title(
        "F1score: {:.3f}".format(f1_score_macro),
        fontsize=22,
    )
    plt.ylabel("Image Label", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.savefig(store_image_path)
    plt.close()

    if return_metrics:
        if fmeasure_alpha:
            return accuracy, f1_score_binary, f1_score_macro, fmeasure02
        else:
            return accuracy, f1_score_binary, f1_score_macro


def create_confusion_matrix_images(result_df, store_image_path, accuracy, fmeasure):
    """
    Create a figure containing images organized in a 2x2 grid based on the confusion matrix.
    Parameters:
    result_df (pd.DataFrame): DataFrame containing the true labels and
                              the high similarity labels.
    store_image_path (str): Path to save the confusion matrix image.
    accuracy (float): Accuracy score.
    fmeasure (float): F1-score.
    Returns:
    None
    """
    # Prepare the dataframes for each quadrant
    tn_df = result_df[(result_df["y_true"] == 0) & (result_df["y_pred_class"] == 0)]
    fp_df = result_df[(result_df["y_true"] == 0) & (result_df["y_pred_class"] == 1)]
    fn_df = result_df[(result_df["y_true"] == 1) & (result_df["y_pred_class"] == 0)]
    tp_df = result_df[(result_df["y_true"] == 1) & (result_df["y_pred_class"] == 1)]

    # Organize the dataframes into a 2x2 grid
    all_df = [[tn_df, fp_df], [fn_df, tp_df]]

    # Determine the number of cells needed for the largest quadrant
    cell_num = int(
        np.max(
            [
                np.ceil(np.sqrt(len(df)))
                for df in [item for sublist in all_df for item in sublist]
            ]
        )
    )

    # Create subplots
    fig, ax = plt.subplots(
        2 * cell_num, 2 * cell_num, figsize=(cell_num * 6, cell_num * 4)
    )

    # Plot the images in each quadrant
    for m in range(2):
        for j in range(2):
            df = all_df[m][j]
            if "num_of_ones" in df.columns:
                df = df.sort_values("num_of_ones")
            else:
                df = df.sort_values("y_pred_class")
            sample_num = len(df)
            index_list = np.arange(sample_num)
            block_array = [[] for _ in range(cell_num)]
            for k in index_list:
                block_array[k % cell_num].append(k)
            for row in block_array:
                while len(row) < cell_num:
                    row.append(None)

            for k, row in enumerate(block_array):
                for p, index in enumerate(row):
                    ax_idx = (m * cell_num + k, j * cell_num + p)
                    if index is None:
                        ax[ax_idx].set_facecolor("white")
                        ax[ax_idx].axis("off")
                        continue
                    img = plt.imread(f"{df.iloc[index]['image_path']}")
                    ax[ax_idx].imshow(img)
                    ax[ax_idx].axis("off")
                    # Display the number of 'Private' features
                    if "num_of_ones" in df.columns:
                        ax[ax_idx].set_title(
                            f"Num of Private: {df.iloc[index]['num_of_ones']}"
                        )
                    else:
                        ax[ax_idx].set_title(
                            f"Predict: {df.iloc[index]['y_pred_class']}"
                        )

    # Positions for vertical line
    left_axes = ax[0, cell_num - 1]
    right_axes = ax[0, cell_num]
    left_pos = left_axes.get_position()
    right_pos = right_axes.get_position()
    x_boundary = (left_pos.x1 + right_pos.x0) / 2

    # Positions for horizontal line
    top_axes = ax[cell_num - 1, 0]
    bottom_axes = ax[cell_num, 0]
    top_pos = top_axes.get_position()
    bottom_pos = bottom_axes.get_position()
    y_boundary = (top_pos.y0 + bottom_pos.y1) / 2 + 0.005

    # Draw dividing lines
    fig.add_artist(
        Line2D(
            [x_boundary, x_boundary],
            [0.1, 0.9],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )
    fig.add_artist(
        Line2D(
            [0.1, 0.9],
            [y_boundary, y_boundary],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )
    fig.add_artist(
        Line2D(
            [0.1, 0.1],
            [0.1, 0.9],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )
    fig.add_artist(
        Line2D(
            [0.9, 0.9],
            [0.1, 0.9],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )
    fig.add_artist(
        Line2D(
            [0.1, 0.9],
            [0.1, 0.1],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )
    fig.add_artist(
        Line2D(
            [0.1, 0.9],
            [0.9, 0.9],
            transform=fig.transFigure,
            color="black",
            linewidth=2,
        )
    )

    # Add labels in the corners with larger font sizes
    fig.text(0.05, 0.95, "TN", ha="left", va="top", fontsize=50, color="red")
    fig.text(0.95, 0.95, "FP", ha="right", va="top", fontsize=50, color="red")
    fig.text(0.05, 0.05, "FN", ha="left", va="bottom", fontsize=50, color="red")
    fig.text(0.95, 0.05, "TP", ha="right", va="bottom", fontsize=50, color="red")

    # Add an overall title to the figure
    fig.suptitle(
        "Accuracy: {:.3f}, F1-score: {:.3f}".format(accuracy, fmeasure), fontsize=70
    )

    # Add x-axis label 'Num of Private'
    # fig.text(0.5, 0.005, 'Num of Private', ha='center', va='bottom', fontsize=40)

    # Add labels 'Public' and 'Private' on the x-axis
    fig.text(0.30, 0.05, "Predicted: Public", ha="center", va="bottom", fontsize=50)
    fig.text(0.70, 0.05, "Predicted: Private", ha="center", va="bottom", fontsize=50)

    # Add labels 'Public' and 'Private' on the y-axis
    fig.text(
        0.07,
        0.70,
        "GT: Public",
        ha="left",
        va="center",
        rotation="vertical",
        fontsize=50,
    )
    fig.text(
        0.07,
        0.30,
        "GT: Private",
        ha="left",
        va="center",
        rotation="vertical",
        fontsize=50,
    )

    fig.savefig(store_image_path)
    plt.close()


def create_logits_distribution(logits, targets, title, save_path):
    bins = np.arange(0, 1.05, 0.05)
    plt.figure(figsize=(8, 6))
    plt.hist(logits[targets == 0], bins=bins, label="normal", alpha=0.5)
    plt.hist(logits[targets == 1], bins=bins, label="abnormal", alpha=0.5)
    plt.xlabel("Logit", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(save_path)
