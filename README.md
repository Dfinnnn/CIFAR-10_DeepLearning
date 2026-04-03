# CIFAR-10 Image Classification CNN

A custom PyTorch Convolutional Neural Network built to classify the CIFAR-10 dataset into 10 categories. 

## Features
* Modular architecture (`config.py`, `engine.py`, `model.py`)
* Custom Learning Rate Scheduler (`StepLR`)
* Dynamic CLI for predictions using `argparse`

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Generate submission: `python predict.py`
