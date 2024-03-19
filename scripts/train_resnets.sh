#!/usr/bin/bash
resnets=( 18 34 50 101 152)

for i in "${resnets[@]}"
do
    python "scripts/train_model.py" $i 2 300 data
done