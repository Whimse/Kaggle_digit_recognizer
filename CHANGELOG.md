Release 1.1
- Completed test results in section [architecture exploration](#architecture-exploration) with metrics provided by Kaggle
- Tested performance of additional network architectures (resnet50, resnet101)
- Added command-line parameter to enable/disable weight decay
- Added Dockerfile to allow running containerized training
- Added augmentation functionality using Kornea

Release 1.0
- Initialized code from notebook codebase at Kaggle: https://www.kaggle.com/code/franklemuchahary/mnist-digit-recognition-using-pytorch/notebook
- Added requirements.txt and README.md files
- Created initial repo, pushed to Github
- Several minor changes (evaluating inference time, tqdm)
- Refactored dataset creation code, training and evaluation functions out of _train_nn.py_
- Refactored test code from _train_nn.py_ into _test_nn.py_
- Added command line parameter parsing
- Added MLflow for managing experiments
- Added scripts to generate ensemble predictions
