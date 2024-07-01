#!/bin/bash

python scripts/train_autoencoder.py data/raw/ .tmpres.txt -d true
python scripts/train_autoencoder.py data/raw/ .tmpres.txt -d false
