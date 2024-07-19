#!/usr/bin/bash

trap "exit" INT

clean_datadirs() {
    rm $1/labeled/images/*.tif
    rm $1/labeled/labels/*.tif
    rm $1/raw/images/*.png
    echo "cleaned $1"
}

# ! rerun with
# resnet 34
# 100 and 20 samples - DONE 5.7.2024
# full sampling for baseline resnets or also reduced sampling

# number setup
#! correct values
n_labeled=100
n_unlabeled=2000
epochs_labeled=200
epochs_unlabeled=300
seed=42
heads=10
name="eccv_retrain"

# ! test values
# n_labeled=5
# n_unlabeled=10
# epochs_labeled=5
# epochs_unlabeled=5
# seed=42
# heads=3
# name="11111"

# dir setup
data_subdir="subensemble"
datadir_ssd="/home/mandel/data_ssd/$name/data/$data_subdir"
