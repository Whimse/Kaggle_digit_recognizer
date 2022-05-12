
# Overview

The code in this repository can be used to train any neural network provided in the [torchvision.models](https://pytorch.org/vision/stable/models.html) package on the MNIST dataset. These networks are trained with a standard configuration of the Adam optimizer. No data augmentation was included in the training.

The repository also includes functionality to produce an ensemble predictor. This ensemble generates outputs combining predictions provided by multiple networks.

The code also contains functionality to produce submissions to the [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) Kaggle competition for these prediction algorithms. The ensemble predictor produced with this code was able to reach [9% top position in the leaderboard of the Kaggle competition](https://www.kaggle.com/reanor/competitions?tab=active).

# Code Usage

## Setting up Work Environment

The code was tested with Python version 3.6.9.

The execution of the training requires several packages, including Pytorch, Pandas or Kaggle. These can be automatically installed with the following command line:

    pip3 install -r requirements.txt --force-reinstall

## Downloading data

Using the kaggle command line tool:

    kaggle competitions download -c digit-recognizer
    unzip -d data digit-recognizer.zip

## Training

The following line will train a network to solve MNIST10, given the name of a network available in the [torchvision.models](https://pytorch.org/vision/stable/models.html) package:

    python3 -u src/train.py --model_name <name> --epochs <num_epochs>

Example:

    python3 -u src/train.py --model_name resnet34 --epochs 10

The following command displays a list of the network architectures available for training:

    python3 -u src/train_nn.py --help

The script _scripts/batch_train.sh_ can be used to train several network architectures sequencially, including Resnet, Mobilenet or EfficientNet.

## Visualizing Results

Each network training done with the _train.py_ script will register results (model, metrics, etc...) in an MLflow experiment, by default inside the _mlruns_ folder.

These experiments can be visualized using the MLflow UI, which can be launched with the following command:

    mlflow ui

Typically, the UI can be accessed by opening the following URL in any browser:

    http://127.0.0.1:5000/

## Generating Kaggle Submissions

After training a model with the _src/train.py_ script, the script _src/generate_submission.py_ can be used to generate the Kaggle submission file _submission.csv_:

    python3 -u src/generate_submission.py <model_file>

The submission files can also be generated in bulk with the script _scripts/parse_results.sh_. It will not only produce submission files for all the networks trained with the _train.py_ script, but it will also produce the predictions for the ensemble method in the file _submission_ensemble.csv_.

# Experimental Results

## Architecture Exploration

The first experiment evaluated the accuracy achievable by different network architectures in the Python model zoo.

Resnet34, the network with top test accuracy, provided a 0.99403 accuracy in the _Digit Recognizer_ Kaggle competition, reaching position 267/2055 (top 12,99%) in the leaderboard.

| Architecture  | Validation Accuracy  | Test Accuracy (Kaggle)&ast;  | Inference time (ms/batch) |
|---                    |---            |---            |---        |
| resnet18              | 0.9945        |               | 0.1555    |
| resnet34              | **0.9950**    | **0.99403**   | 0.2880    |
| mobilenet_v2          | 0.9949        |               | 0.1907    |
| mobilenet_v3_small    | 0.9815        |               | 0.1742    |
| mobilenet_v3_large    | 0.9929        | 0.99060       | 0.2421    |
| squeezenet1_0         | 0.9933        |               | **0.0913**|
| squeezenet1_1         | 0.9886        |               | 0.1010    |
| efficientnet_b0       | 0.9935        |               | 0.3589    |
| efficientnet_b3       | 0.9905        |               | 0.6917    |
| mnasnet0_5            | 0.7882        | 0.78892       | 0.1429    |
| mnasnet1_0            | 0.9876        |               | 0.2218    |

&ast; Some of the fields in this column could not be completed due to submission limit per day imposed by Kaggle.

## Ensemble Prediction

The second experiment evaluated the performance of a simple ensemble predictor. For a given test sample, the ensemble yields the most frequent prediction given by the following networks:
- resnet18
- resnet34
- mobilenet_v2
- mobilenet_v3_large
- squeezenet1_0
- efficientnet_b0

This ensemble method yielded a 0.99535 accuracy, which reached position 176/2055 (top 8,56%) in the Kaggle leaderboard.

The added inference time for this predictor would be 1,1358ms per batch, according to the results in the table above.

## Conclusions

The ensemble method would be the best choice for applications where accuracy is the crucial factor. Meanwhile, for applications where prediction time is also a crucial factor, a network like _squeezenet_1_0_ can provide very competitive results compared to the performance of the ensemble.

The accuracy for the ensemble is just 2% higher than the accuracy for squeezenet 1.0. Meanwhile, the processing time per batch is 12.4 times larger.

## Further Experiments

The first experiment tried to optimize the accuracy by exploring different choices for the network architecture. Additional network architectures could have been considered for the experiments. Additionally, other hyperparameters such as the optimization method, or the learning rate, could have been optimized in the experiments. Training longer with SGD with a more conservative LR decay or larger batch sizes can typically provide improvements in accuracy.

Data augmentation was not used in these experiments, and it is a very effective way to [further increase the accuracy with image classification problems](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/).

[More sophisticated approaches](https://paperswithcode.com/sota/image-classification-on-mnist) have been also proposed in the academic literature, reaching accuracies in the range of 0,9991 for the digit classification problem with MNIST. Any of these ideas could be explored to further push the prediction accuracy of the models.
