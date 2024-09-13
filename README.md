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

This repository implements a Federated Learning pipeline using the Flower framework, where multiple clients collaboratively train a model on the MNIST dataset for image classification. Clients train on their local data partitions without sharing raw data, preserving privacy.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, we implement a simple Federated Learning pipeline for image classification using the MNIST dataset. The pipeline includes:

  - **`client.py`:** Defines the Flower client to handle local training on partitioned MNIST data for each client.
  - **`server.py`:** Defines server-side configurations for managing communication rounds, model aggregation, and global model updates.
  - **`dataset.py`:** Handles loading the MNIST dataset, strategically partitioning it among clients (e.g., IID or non-IID), and ensures no raw data is exchanged.
  - **`model.py`:** Defines a simple Convolutional Neural Network (CNN) model for image classification.
  - **`main.py`:** Orchestrates the federated learning simulation using Flower, coordinating communication between clients and the server.

## Installation

### Prerequisites

- Python 3.9 or later
- [Flower](https://flower.dev/)
- [PyTorch](https://pytorch.org/)

### Install Dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
