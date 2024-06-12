#!/usr/bin/bash
dss=("flowering" "vegetative" "combined")
resnets=("ae/RN18-32.ckpt" "ae/RN18-256.ckpt")

for ds in "${dss[@]}"
do  
    for i in "${resnets[@]}"
    do
        python "scripts/train_from_ae.py" 18 1 200 $ds $i  --limit 0.01
    done 
done

for ds in "${dss[@]}"
do  
    python "scripts/train_from_ae.py" 34 1 200 $ds "ae/RN34-32.ckpt" --limit 0.01
done
