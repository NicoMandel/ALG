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

# Cleaning before a new run!
clean_datadirs $datadir_ssd

# run subensemble with normal autoencoder
python scripts/subensemble_pipeline.py $name $data_subdir --autoenc --retrain  --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with normal autoencoder and ensemble sampling"
clean_datadirs $datadir_ssd
python scripts/subensemble_pipeline.py $name $data_subdir --autoenc --retrain  --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with normal autoencoder and without sampling"
clean_datadirs $datadir_ssd

# run subensemble with denoising autoencoder
python scripts/subensemble_pipeline.py $name $data_subdir --autoenc --denoising --retrain --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with denoising autoencoder and sampling"
clean_datadirs $datadir_ssd
python scripts/subensemble_pipeline.py $name $data_subdir --autoenc --denoising --retrain --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with denoising autoencoder without sampling"
clean_datadirs $datadir_ssd
