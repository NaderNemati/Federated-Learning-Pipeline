# Federated-Learning-Pipeline
Federated Learning pipeline using Flower and PyTorch


## Project Structure

The project is organized into the following directories:

- **client.py:** Handles local training on MNIST data partitions for each client.
- **dataset.py:** Partitions the MNIST dataset among clients.
- **main.py:** Orchestrates the federated learning simulation using Flower.
- **model.py:** Defines the CNN model for image classification.
- **server.py:** Defines server-side configurations for federated learning rounds.
- **conf/base.yaml:** Contains configuration options for the simulation (default configuration file).
- **requirements.txt:** Lists the required dependencies for the project.
- **README.md:** Provides documentation for the project (you're currently reading it!).
- **LICENSE:** Specifies the project's licensing information.






# Federated Learning Pipeline on MNIST Dataset

This repository implements a Federated Learning pipeline using the Flower framework, where multiple clients collaboratively train a model on the MNIST dataset for image classification.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, we implement a simple Federated Learning pipeline for image classification using the MNIST dataset. The pipeline includes:
- A `client.py` file, which defines the Flower client to handle local training on MNIST data.
- A `server.py` file, defining server-side configurations for federated learning rounds.
- A `dataset.py` file for partitioning the MNIST dataset among clients.
- A `model.py` file, defining a simple Convolutional Neural Network (CNN) model for classification.
- A `main.py` file to orchestrate the federated learning simulation using Flower.

## Installation

### Prerequisites
- Python 3.9 or later
- [Flower](https://flower.dev/)
- [PyTorch](https://pytorch.org/)

### Install dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
