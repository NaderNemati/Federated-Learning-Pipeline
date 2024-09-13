# Federated-Learning-Pipeline

This repository implements a Federated Learning pipeline using Flower and PyTorch, enabling collaborative training of a model on the MNIST dataset for image classification across multiple clients without sharing raw data.

## Project Structure (Visualized)

<div align="center">
  <table border=0 style="border: 1.2px solid #c6c6c6 !important; border-spacing: 2px; width: auto !important;">
    <tr>
      <td valign=top style="border: 1.2px solid #c6c6c6 !important; padding: 2px !important;">
        <div align=center valign=top>
          <img src="https://raw.githubusercontent.com/NaderNemati/Federated-Learning-Pipeline/main/federated-learning-pipeline.png" alt="Project Structure" style="margin: 0px !important; height: 200px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

## Project Structure
### Federated-Learning-Pipeline
```python
  ├── client.py        # Client-side code for federated learning with Flower
  ├── dataset.py       # Dataset loading, partitioning, and distribution
  ├── main.py          # Main entry point for running the simulation
  ├── model.py         # Model definition (CNN for MNIST)
  ├── server.py        # Server-side federated learning configuration
  ├── requirements.txt # Dependencies to run the code
  ├── README.md        # Project documentation (you're here!)
  ├── LICENSE          # License file
  └── conf/            # Configuration files for the simulation (base.yaml)




## Federated Learning Pipeline on MNIST Dataset

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

    └── base.yaml    # Configuration files for the simulation


