#!/usr/bin/bash

trap "exit" INT

clean_datadirs() {
    find $1/labeled/images/ -maxdepth 1 -mindepth 1 -type f -name "*.tif" -delete 
    find $1/labeled/labels/ -maxdepth 1 -mindepth 1 -type f -name "*.tif" -delete 
    find $1/raw/images/ -maxdepth 1 -mindepth 1 -type f -name "*.png" -delete 
    echo "cleaned $1"
}

# ! rerun with
# resnet 34
# 100 and 20 samples - DONE 5.7.2024
# full sampling for baseline resnets or also reduced sampling

# !correct values
n_labeled=100
n_unlabeled=2000
epochs_labeled=200
epochs_unlabeled=500
seed=42
name="eccv_v400"

# number setup
#! test values
# n_labeled=10
# n_unlabeled=20
# epochs_labeled=5
# epochs_unlabeled=5
# seed=42
# name="eccv_v300"

# dir setup
data_subdir="baseline"
datadir_ssd="/home/mandel/data_ssd/$name/data/$data_subdir"


# Cleaning before a new run!
clean_datadirs $datadir_ssd

# run baseline - autoencoder
python scripts/baseline_ae.py $name $data_subdir --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs $datadir_ssd
# with full datasets
python scripts/baseline_ae.py $name $data_subdir --full --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs $datadir_ssd

# run baseline - denoising autoencoder
python scripts/baseline_ae.py $name $data_subdir --denoising --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs $datadir_ssd
# with full datasets
python scripts/baseline_ae.py $name  $data_subdir --denoising --full --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs $datadir_ssd

# baseline - no autoencoder just a resnet
python scripts/baseline_resnet.py $name $data_subdir --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed
clean_datadirs $datadir_ssd
# with full datasetets
python scripts/baseline_resnet.py $name $data_subdir --full --epochs_labeled $epochs_labeled --seed $seed
clean_datadirs $datadir_ssd