<div align="center">
  <table border=0 style="border: 0.2px solid #c6c6c6 !important; border-spacing: 4px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0.2px solid #c6c6c6 !important; padding: 4px !important;">
        <div align=center valign=top>
          <img src="https://raw.githubusercontent.com/NaderNemati/Federated-Learning-Pipeline/main/images/federated-learning-pipeline.png" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>


# Federated-Learning-Pipeline

Federated Learning (FL) is a distributed machine learning approach that allows multiple clients (such as mobile devices or institutions) to collaboratively train a shared model without sharing raw data. This approach is particularly useful in scenarios where data privacy is crucial, such as in healthcare or finance, where sharing sensitive information is not permissible.

## Federated Learning Pipeline on MNIST and CIFAR10

This repository implements a Federated Learning pipeline using the Flower framework with PyTorch, designed for collaborative training of image classification models on the MNIST and CIFAR10 datasets. The pipeline enables multiple clients to train models on their local data partitions, ensuring privacy by never sharing raw data with a central server. Instead, only model updates (weights) are exchanged, and the central server aggregates these updates using the Federated Averaging (FedAvg) algorithm to update a global model. The global model is then distributed back to clients for further training in subsequent rounds. This decentralized approach is crucial in scenarios where data privacy is a priority, such as healthcare or finance.



## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#Configuration)
- [Performance Summary and Analysis](#Performance-Summary-and-Analysis)
- [License](#license)

## Overview

This project implements a simple Federated Learning pipeline for image classification using the MNIST dataset. The pipeline includes:

* **Client-side functionality:**
    * `client.py`: Defines the Flower client to handle local training on partitioned MNIST data for each client.
* **Server-side configuration:**
    * `server.py`: Defines server-side configurations for managing communication rounds, model aggregation, and global model updates.
* **Data handling:**
    * `dataset.py`: Handles loading the MNIST dataset, strategically partitioning it among clients (e.g., IID or non-IID), and ensures no raw data is exchanged.
* **Model definition:**
    * `model.py`: Defines a simple Convolutional Neural Network (CNN) model for image classification.
* **Simulation orchestration:**
    * `main.py`: Orchestrates the federated learning simulation using Flower, coordinating communication between clients and the server.

## Project Structure:

```python
Federated-Learning-Pipeline/
  ├── client.py        # Client-side code for federated learning with Flower
  ├── dataset.py       # Dataset loading, partitioning, and distribution
  ├── main.py          # Main entry point for running the simulation
  ├── model.py         # Model definition (CNN for MNIST)
  ├── server.py        # Server-side federated learning configuration
  ├── requirements.txt # Dependencies to run the code
  ├── README.md        # Project documentation (you're here!)
  ├── LICENSE          # License file
  └── conf/            # Configuration files for the simulation (base.yaml)
    └── base.yaml
```


## Installation

### Prerequisites

- [Python 3.9 or later](https://www.python.org/downloads/)
- [Flower](https://flower.dev/)
- [PyTorch](https://pytorch.org/)

### Install Dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```


## Usage
### Running the Federated Learning Simulation
Clone the repository:

```bash
git clone https://github.com/NaderNemati/Federated-Learning-Pipeline.git
cd Federated-Learning-Pipeline
```

Install dependencies: Make sure the virtual environment is activated if you use one, and run:

```bash
pip install -r requirements.txt
```

Run the simulation: To start the federated learning simulation, run the main.py script:

```bash
python main.py
```

This command orchestrates the federated learning process by:

* Loading and partitioning the MNIST dataset using dataset.py
* Running the federated simulation with client-side logic in client.py and server-side logic in server.py
* Training a CNN model defined in model.py on the partitioned dataset.




## Configuration

You can modify the federated learning simulation parameters by editing the conf/base.yaml file.

### Example for base.yaml content

```python
num_rounds: 2        # Number of communication rounds between clients and the server
num_clients: 5       # Total number of clients
batch_size: 32       # Batch size for local training
num_classes: 10      # Number of output classes (e.g., for MNIST or CIFAR10)
num_clients_per_round_fit: 5  # Number of clients participating in training per round
num_clients_per_round_eval: 5 # Number of clients participating in evaluation per round

config_fit: 
  lr: 0.001           # Learning rate for local training
  weight_decay: 1e-4  # Weight decay to avoid overfitting
  momentum: 0.9       # Momentum for optimization
  local_epochs: 2     # Number of local training epochs per client

dp:
  epsilon: 1.0        # Privacy budget for Differential Privacy
  delta: 1e-5         # Privacy parameter controlling risk of leakage
  noise_scale: 0.01   # Amount of noise added to model updates for DP

```







## Performance Summary and Analysis
In this federated learning experiment, the training and evaluation processes were distributed across 5 clients, each holding a partition of the MNIST dataset. The global model was updated through an aggregation of the models trained locally by the clients. Below is a detailed analysis of the client's performance and the overall learning process:

#### Clients' Loss and Accuracy
* Clients exhibited varying performance due to differences in their local data partitions.
* At the beginning of the training process (Round 1), the loss values across the clients were relatively high, and accuracy ranged between 75-82%.
* As training progressed, clients steadily improved in both loss and accuracy. For instance, by Round 5, client accuracies reached about 95-96%.
* By Round 10, clients achieved accuracies between 96.5% and 97.5%, with losses as low as 4.6-5.8.

#### Global Model Performance
* The global model initially showed poor performance, with low accuracy (around 75%) and high loss.
* By Round 5, the global model's accuracy improved to 96.6%, with a loss of 6.85.
* By Round 10, the global model achieved an accuracy of 97.5%, and the loss decreased to 4.65.

#### Comparison of Clients and Global Model Performance
* Clients generally outperformed the global model in accuracy, as they optimized for their specific local data.
* The global model, however, was more generalized, performing well across all clients and unseen data.
* The global model showed slower but more consistent improvements compared to individual clients.

#### Learning Process of the Global Model
* **Initial Stages (Rounds 1-3):** The global model made rapid progress in reducing loss and improving accuracy, as it learned from the distributed data across the clients.
* **Mid-Training (Rounds 4-7):** The global model continued to improve but at a slower rate as the clients' models started to converge.
* **Final Rounds (Rounds 8-10):** The global model approached convergence, with incremental improvements in accuracy and loss.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the [LICENSE](LICENSE) file for more details.
