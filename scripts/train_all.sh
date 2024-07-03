#!/usr/bin/bash

trap "exit" INT

clean_datadirs() {
    rm /home/mandel/data_ssd/v2/data/labeled/images/*.tif
    rm /home/mandel/data_ssd/v2/data/labeled/labeled/*.tif
    rm /home/mandel/data_ssd/v2/data/raw/images/*.png
    echo "cleaned /home/mandel/data_ssd/v2/data/"
}

# number setup
n_labeled=100
n_unlabeled=20
epochs_labeled=5
epochs_unlabeled=5
seed=42
heads=10
name="eccv_v2"

# Cleaning before a new run!
clean_datadirs

# run subensemble with normal autoencoder
python scripts/subensemble_pipeline.py $name --autoenc --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with normal autoencoder and ensemble sampling"
clean_datadirs
python scripts/subensemble_pipeline.py $name --autoenc --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with normal autoencoder and without sampling"
clean_datadirs

# run subensemble with denoising autoencoder
python scripts/subensemble_pipeline.py $name --autoenc --denoising --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with denoising autoencoder and sampling"
clean_datadirs
python scripts/subensemble_pipeline.py $name --autoenc --denoising --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed --heads $heads
echo "completed running subensemble with denoising autoencoder without sampling"
clean_datadirs

# run subensemble without autoencoder training at all
python scripts/subensemble_pipeline.py $name --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed --heads $heads
echo "completed running subensemble without autoencoder with sampling"
clean_datadirs
python scripts/subensemble_pipeline.py $name --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed --heads $heads
echo "completed running subensemble without autoencoder and without sampling"
clean_datadirs

# run baseline - autoencoder
python scripts/baseline_ae.py $name --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs

# run baseline - denoising autoencoder
python scripts/baseline_ae.py $name --denoising --n_labeled $n_labeled --epochs_labeled $epochs_labeled --epochs_unlabeled $epochs_unlabeled --n_unlabeled $n_unlabeled --seed $seed
clean_datadirs

# baseline - no autoencoder just a resnet
python scripts/baseline_resnet.py $name --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed

