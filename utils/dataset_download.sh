#!/bin/bash

CURRENT_DIR=$(pwd)
ROOT_DIR=$CURRENT_DIR/datasets/ucf_sports_actions

# wget --no-check-certificate -P $ROOT_DIR https://www.crcv.ucf.edu/data/ucf_sports_actions.zip
# Note: The train_test_split URL may be outdated - check if file exists locally or find alternative source
# https://github.com/hakimnasaoui/Human-Action-Recognition-UCF-Sport/tree/master
# wget -P $ROOT_DIR http://www.sfu.ca/~tla58/other/train_test_split


# unzip "$ROOT_DIR/ucf_sports_actions.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/ucf_sports_actions.zip"

# mv $ROOT_DIR/ucf_sports_actions/*/ $ROOT_DIR/
# rm -r $ROOT_DIR/ucf_sports_actions

# find "$ROOT_DIR" -name "*.jpg" -type f -delete

ROOT_DIR=$CURRENT_DIR/datasets/sv_wild
# wget -P $ROOT_DIR https://www.cse.msu.edu/computervision/SVW.zip

# unzip "$ROOT_DIR/SVW.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/SVW.zip"

uv run python utils/svw_prepare.py --csv $ROOT_DIR/SVW.csv --output $ROOT_DIR --split 1
