# Neural Network Classifiers – CIFAR-10

This project implements and compares the performance of three neural network models on the CIFAR-10 dataset:

- **MLP (Multi-layer Perceptron)**
- **CNN (Convolutional Neural Network)**
- **RESNET-like Custom Architecture**

## Setup 
1. run `make venv`
2. run `. venv/bin/activate`
3. run `make install`

## Usage
run `python <script>`
where `<script>` = [MLP.py, CNN.py, RESENT.py]

MODEL PERFORMANCE
-----------------
Each model has been parameter tuned so that they reach a certain performance. 
Note the use of a seed when generating random weights have been used to ensure that 
the parameters can be more reliably and specifically tuned. Listed here are the maximum performances'
each model attains throughout the 15 training epochs
