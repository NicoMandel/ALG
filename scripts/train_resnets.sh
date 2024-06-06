#!/usr/bin/bash
resnets=(34 50 101)
dss=("flowering" "vegetative" "combined")

for i in "${resnets[@]}"
do
    for ds in "${dss[@]}"
    do
        python "scripts/train_model.py" $i 1 200 $ds -tr 
    done 
done
