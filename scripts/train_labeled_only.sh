#!/usr/bin/bash

trap "exit" INT

clean_datadirs() {
    rm /home/mandel/data_ssd/no_ae/data/labeled/images/*.tif
    rm /home/mandel/data_ssd/no_ae/data/labeled/labeled/*.tif
    echo "cleaned /home/mandel/data_ssd/no_ae/data/"
}

# number setup
n_labeled=20
epochs_labeled=200
seed=42
name="eccv_no_ae"

# Cleaning before a new run!
clean_datadirs

# run subensemble without autoencoder training at all
python scripts/subensemble_pipeline.py $name --sample --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed
echo "completed running subensemble without autoencoder with sampling"
clean_datadirs
python scripts/subensemble_pipeline.py $name --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed
echo "completed running subensemble without autoencoder and without sampling"
clean_datadirs

# baseline - no autoencoder just a resnet
python scripts/baseline_resnet.py $name --n_labeled $n_labeled --epochs_labeled $epochs_labeled --seed $seed

