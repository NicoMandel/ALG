#!/usr/bin/bash
resnets=(18 34 50 101 152)
trto=("" "-tr" "-tr -to")

for i in "${resnets[@]}"
do
    for tr in "${trto[@]}"
    do
        echo "scripts/train_model.py" $i 2 200 data $tr 
    done 
done
