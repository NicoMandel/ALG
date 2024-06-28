#!/usr/bin/bash
clean_datadirs() {
    rm /home/mandel/data_ssd/ALG/labeled/images/*.tif
    rm /home/mandel/data_ssd/ALG/labeled/labeled/*.tif
    rm /home/mandel/data_ssd/ALG/raw/images/*.png
    echo "cleaned /home/mandel/data_ssd/ALG/"
}

# Cleaning before a new run!
clean_datadirs

# run subensemble
python scripts/subensemble_pipeline.py true
echo "completed running subensemble with sampling"
clean_datadirs
python scripts/subensemble_pipeline.py false
echo "completed running subensemble without sampling"
clean_datadirs

# run baseline - autoencoder
python scripts/baseline_ae.py
clean_datadirs
# baseline - no autoencoder
python scripts/baseline_resnet.py

