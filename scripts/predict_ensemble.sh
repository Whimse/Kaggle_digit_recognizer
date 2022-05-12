#!/bin/bash

# Generate Kaggle submission files for experiments
for EXPERIMENT in mlruns/0/*/; do
    CODE=$EXPERIMENT
    CODE=${CODE#mlruns/0/}
    CODE=${CODE:0:10}
    NAME=`cat $EXPERIMENT/tags/mlflow.runName`
    INFERENCE_TIME=`python src/generate_submission.py "$EXPERIMENT"artifacts/model/data/model.pth 2>/dev/null | grep -e Inference | cut -d ":" -f 4`
    echo "$CODE"_"$NAME" inference time: $INFERENCE_TIME
    mv submission.csv submission_"$CODE"_"$NAME".csv
done

# Produce ensemble prediction
python src/ensemble_predictions.py submission_*_mobilenet_v2.csv submission_*_resnet34.csv submission_*_resnet18.csv submission_*_squeezenet1_0.csv submission_*_efficientnet_b0.csv submission_*_mobilenet_v3_large.csv
