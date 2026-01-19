#!/bin/bash
# Check if the correct number of arguments are provided
if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <img_dataset> <txt_dataset> <types> <img_split>[--knn_regressor] [--multi_knn]"
    exit 1
fi

# Assign command line arguments to variables
IMG_DATASET=$1
TXT_DATASET=$2
TYPE=$3
KNN_METRICS=$4
IMG_SPLIT=$5

# Define the base command
BASE_CMD="python clipknn/clip_knn.py --img_dataset $IMG_DATASET --txt_dataset $TXT_DATASET --type $TYPE --img_split $IMG_SPLIT"

# Check if the fourth argument is provided and if it is --knn_regressor
if [ "$#" -eq 6 ] && [ "$6" == "--knn_regressor" ]; then
    BASE_CMD="$BASE_CMD --knn_regressor"
fi
# Check if the fourth or fifth argument is provided and if it is --multi_knn
if [ "$#" -eq 6 ] && [ "$6" == "--multi_knn" ]; then
    BASE_CMD="$BASE_CMD --multi_knn"
elif [ "$#" -eq 7 ]; then
    if [ "$6" == "--knn_regressor" ] && [ "$7" == "--multi_knn" ]; then
        BASE_CMD="$BASE_CMD --knn_regressor --multi_knn"
    elif [ "$6" == "--multi_knn" ] && [ "$7" == "--knn_regressor" ]; then
        BASE_CMD="$BASE_CMD --multi_knn --knn_regressor"
    fi
fi

CONFIG_PATH="configs/clip_model"

# Define the different configurations
# CONFIGS=("$CONFIG_PATH/lptdataset_default.yaml" "$CONFIG_PATH/lptdataset_base-16.yaml" "$CONFIG_PATH/lptdataset_large-14.yaml")
CONFIGS=("$CONFIG_PATH/privacy_default.yaml" "$CONFIG_PATH/privacy_base-16.yaml" "$CONFIG_PATH/privacy_large-14.yaml")

# KNN_WEIGHTS=("uniform" "distance")
KNN_WEIGHTS=("uniform")
# K_VALUES=(1 3 5 7 9)
K_VALUES=(7)
# KNN_ALGORITHMS=("ball_tree" "kd_tree" "brute")
# KNN_ALGORITHMS=("brute")
KNN_ALGORITHMS=("kd_tree")

# Loop through each configuration and k value, and execute the command
for CONFIG in "${CONFIGS[@]}"; do
    for KNN_WEIGHT in "${KNN_WEIGHTS[@]}"; do
        for K in "${K_VALUES[@]}"; do
            for KNN_METRIC in "${KNN_METRICS[@]}"; do
                for KNN_ALGORITHM in "${KNN_ALGORITHMS[@]}"; do
                    CMD=" $BASE_CMD -c $CONFIG --knn_weights $KNN_WEIGHT --knn_metric $KNN_METRIC --k $K --knn_algorithm $KNN_ALGORITHM"
                    echo "Executing: $CMD"
                    OUTPUT=$($CMD)
                    
                    # Extract accuracy and fscore from the output
                    ACCURACY=$(echo "$OUTPUT" | grep "Accuracy" | awk '{print $2}')
                    FSCORE=$(echo "$OUTPUT" | grep "F1-score" | awk '{print $2}')
                done
            done
        done
    done
done
