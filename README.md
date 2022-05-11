
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


# Results

