#!/bin/bash

NETS='resnet18 resnet34 mobilenet_v2 mobilenet_v3_small mobilenet_v3_large efficientnet_b0 efficientnet_b3 squeezenet1_0 squeezenet1_1 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3'

# Run one experiment at a time (requires task spooler installed)
tsp -S 1

# Queue experiments
for NET in $NETS; do
    tsp python3 -u src/train.py --augment --model_name $NET
    tsp python3 -u src/train.py --model_name $NET
done
