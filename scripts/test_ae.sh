#!/bin/bash

python scripts/train_autoencoder.py data/raw/ .tmpres.txt -d -n 2000
python scripts/train_autoencoder.py data/raw/ .tmpres.txt -n 2000
