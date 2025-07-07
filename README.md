# Machine Learning Project Suite

This repository contains a collection of supervised and reinforcement learning models built using PyTorch. It is structured as an educational and experimental platform for exploring key ML paradigms, including neural image classification and reinforcement learning in discrete environments.

## Overview

The project is divided into two main components:

### 1. **Supervised Learning: CIFAR-10 Image Classification**

Implements and compares three neural network architectures trained on the CIFAR-10 dataset:

- **MLP (Multi-Layer Perceptron)** — A fully connected baseline for classification tasks.
- **CNN (Convolutional Neural Network)** — A standard architecture leveraging spatial hierarchies in images.
- **Custom ResNet-like Model** — A deeper network with residual connections to improve training stability and accuracy.

---

### 2. **Reinforcement Learning: GridWorld Agent Training**

Implements and evaluates reinforcement learning agents in GridWorld-style environments inspired by the classic FourRooms benchmark.

- **Environments:** Custom scenarios with varied layouts and terminal conditions
- **Agents:** Tabular Q-learning and Deep Q-Network (DQN) agents
- **Exploration Strategies:** Epsilon-greedy with decay
- **Evaluation:** Tracks episodic returns and learning curves

#### Screenshot of RL agent
