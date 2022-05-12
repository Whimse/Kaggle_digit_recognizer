
# Overview

# Usage

## Download data

Using the kaggle command line tool:

    kaggle competitions download -c digit-recognizer
    unzip -d data digit-recognizer.zip

## Train from host

### Install required packages

Install neccessary packages:

    pip3 install -r requirements.txt --force-reinstall

### Launch network training

The following line will train a network to solve MNIST10:

    python3 -u src/train_nn.py --model_name <name> --epochs <num_epochs>

Example:

    python3 -u src/train_nn.py --model_name resnet34 --epochs 10

The script can train any model architecture available at _torchvision.models_. To see a list of models available, please use the following command line:

    python3 -u src/train_nn.py --help

The training will generate the file name _models/<name>.pt_ containing the model.

### Launch MLflow interface

    mlflow ui

## Train from Docker container

Launch batch training:

    docker build -t docker-ml-model -f Dockerfile .

Retrieve trained networks:

    ...

## Generate Kaggle submission

After training a model with _train_nn.py_, the script _test_nn.py_ can be used to generate the Kaggle submission file

    python3 -u src/test_nn.py <model_file>


# Experimental Results

## Architecture Selection

The first experiment explored the accuracy achievable by different network architectures in the Python model zoo.

Resnet34, the network with top test accuracy, provided a 0.99403 accuracy, reaching position 267/2055 (top 12,99%) in the Kagge leaderboard for the _Digit Recognizer_ competition.

| Architecture  | Validation Accuracy  | Test Accuracy (Kaggle)&ast;  | Inference time (ms/batch) |
|---                    |---        |---|---|
| resnet18              | 0.607     |           | 0.1555    |
| resnet34              | *0.536*   | 0.99403   | 0.2880    |
| mobilenet_v2          | *0.536*   |           | 0.1907    |
| mobilenet_v3_small    | 1.893     |           | 0.1742    |
| mobilenet_v3_large    | 0.738     | 0.99060   | 0.2421    |
| squeezenet1_0         | 0.667     |           | 0.0913    |
| squeezenet1_1         | 1.19      |           | *0.1010*  |
| efficientnet_b0       | 0.655     |           | 0.3589    |
| efficientnet_b3       | 1.024     |           | 0.6917    |
| mnasnet0_5            | 21.18     | 0.78892   | 0.1429    |
| mnasnet1_0            | 1.238     |           | 0.2218    |

&ast; Some of the fields in this column could not be completed due to submission limit per day imposed by Kaggle.


## Ensemble Prediction

The second experiment explored the performance of a simple ensemble predictor. For a given test sample, this method produces the most frequent prediction provided by the following networks:
- mobilenet_v2
- resnet18
- resnet34
- squeezenet1_0
- efficientnet_b0
- mobilenet_v3_large
This provided a 0.99535 accuracy, scaling up to position 176/2055 (top 8,56%) in the Kaggle leaderboard.

## Further Experiments

The first experiment tried to optimize the accuracy by exploring different choices for the network architecture. Additional network architectures could have been considered for the experiments. 
Furthermore, other hyperparameters such as the optimization method or the learning rate could have been optimized in the experiments. Training longer with SGD with a more conservative LR decay or larger batch sizes can typically provide improvements in accuracy.

Data augmentation was not used in these experiments, and it is a very effective way to [further increase the accuracy with image classification problems](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/).

[More sophisticated approaches](https://paperswithcode.com/sota/image-classification-on-mnist) have been also proposed in the academic literature, reaching accuracies in the range of 0,9991 for the digit classification problem with MNIST. Any of these ideas could be explored to further push the prediction accuracy of the models.
