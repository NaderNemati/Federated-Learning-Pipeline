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

- [Project Overview](#Project_Overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modular and Configurable](#Modular_and_Configurable)
- [Performance Summary and Analysis](#Performance_Summary_and_Analysis)
- [License](#license)



## Project Overview

This project implements a Federated Learning (FL) pipeline using the Flower framework and PyTorch for image classification on the MNIST and CIFAR10 datasets. By using the pipeline, multiple clients can collaborate on training a global model while maintaining privacy. Clients share updated models instead of raw data, which are aggregated by a central server using Federated Averaging (FedAvg). The decentralized training approach is ideal for scenarios where data privacy is critical, such as healthcare and finance.

### Key Features

**1-Client-Side Training with Privacy Preservation:**

Clients train models locally on their data partitions and only send model updates (weights) to the server, ensuring that raw data remains private. 
Differential Privacy (DP) adds noise to the model updates, protecting sensitive information, while Secure Aggregation ensures that individual updates 
remain private during the aggregation process.
      
**2-Training and Validation Strategy:**

Data is split IID (Independent and Identically Distributed) among clients, meaning each client’s data is representative of the entire dataset. This method leverages the fact that        both MNIST and CIFAR10 datasets are drawn from a common distribution. Additionally, 10% of each client's data is reserved for local validation, allowing clients to assess their          model's performance during training. After every round of training and aggregation, the global model is evaluated on the full test dataset, ensuring comprehensive performance            analysis.

**3-Dynamic Model Selection:**

The project supports both MNIST and CIFAR10 datasets, with models that are tailored to the complexity of each dataset:

**MNISTNet:** A relatively simple model with two convolutional layers, suited for MNIST's grayscale images. It efficiently handles the lower complexity of MNIST data, which            has a single color channel.

| Layer Type           | Input Shape       | Output Shape        | Kernel Size | Stride | Padding | Additional Info                  |
|----------------------|-------------------|---------------------|-------------|--------|---------|----------------------------------|
| Conv2D               | (1, 28, 28)       | (32, 24, 24)        | 5x5         | 1      | 0       | 1 input channel (grayscale)      |
| MaxPool2D            | (32, 24, 24)      | (32, 12, 12)        | 2x2         | 2      | 0       | Pooling layer                    |
| Conv2D               | (32, 12, 12)      | (64, 8, 8)          | 5x5         | 1      | 0       |                                  |
| MaxPool2D            | (64, 8, 8)        | (64, 4, 4)          | 2x2         | 2      | 0       | Pooling layer                    |
| Flatten              | (64, 4, 4)        | (1024)              | N/A         | N/A    | N/A     | Converts 2D to 1D                |
| Fully Connected (FC) | (1024)            | (120)               | N/A         | N/A    | N/A     | Fully connected layer            |
| Fully Connected (FC) | (120)             | (84)                | N/A         | N/A    | N/A     | Fully connected layer            |
| Fully Connected (FC) | (84)              | (10)                | N/A         | N/A    | N/A     | Output layer (10 classes)        |


**CIFAR10Net:** A more complex model designed for RGB images in CIFAR10. It features deeper layers, batch normalization, and dropout to handle the higher complexity and                     variability of CIFAR10 data. Additionally, gradient clipping is used to prevent exploding gradients during training, a critical feature for more complex datasets like                    CIFAR10.

| Layer Type           | Input Shape       | Output Shape        | Kernel Size | Stride | Padding | Additional Info                  |
|----------------------|-------------------|---------------------|-------------|--------|---------|----------------------------------|
| Conv2D               | (3, 32, 32)       | (64, 32, 32)        | 3x3         | 1      | 1       | 3 input channels (RGB)           |
| BatchNorm2D          | (64, 32, 32)      | (64, 32, 32)        | N/A         | N/A    | N/A     | Batch normalization for stability|
| MaxPool2D            | (64, 32, 32)      | (64, 16, 16)        | 2x2         | 2      | 0       | Pooling layer                    |
| Conv2D               | (64, 16, 16)      | (128, 16, 16)       | 3x3         | 1      | 1       |                                  |
| BatchNorm2D          | (128, 16, 16)     | (128, 16, 16)       | N/A         | N/A    | N/A     | Batch normalization for stability|
| MaxPool2D            | (128, 16, 16)     | (128, 8, 8)         | 2x2         | 2      | 0       | Pooling layer                    |
| Conv2D               | (128, 8, 8)       | (256, 8, 8)         | 3x3         | 1      | 1       |                                  |
| BatchNorm2D          | (256, 8, 8)       | (256, 8, 8)         | N/A         | N/A    | N/A     | Batch normalization for stability|
| MaxPool2D            | (256, 8, 8)       | (256, 4, 4)         | 2x2         | 2      | 0       | Pooling layer                    |
| Flatten              | (256, 4, 4)       | (4096)              | N/A         | N/A    | N/A     | Converts 2D to 1D                |
| Fully Connected (FC) | (4096)            | (512)               | N/A         | N/A    | N/A     | Fully connected layer            |
| Dropout (0.5)        | (512)             | (512)               | N/A         | N/A    | N/A     | Dropout to prevent overfitting   |
| Fully Connected (FC) | (512)             | (256)               | N/A         | N/A    | N/A     | Fully connected layer            |
| Fully Connected (FC) | (256)             | (10)                | N/A         | N/A    | N/A     | Output layer (10 classes)        |


**4-Modular and Configurable:**

The project is structured into clean, modular components:

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

###### Dataset Preparation (dataset.py):

This script loads and partitions the MNIST and CIFAR10 datasets, applying transformations like normalization and ensuring that data is split among clients. Each client trains its model on its data without sharing it, adhering to federated learning principles.

###### Model Definition (model.py):

Defines flexible model architectures for MNIST and CIFAR10. Based on the selected dataset, an appropriate model is dynamically initialized. This script also handles optimization, loss calculation, and evaluation, enabling seamless switching between datasets and models.

###### Client Setup (client.py):

Manages local training and evaluation on each client through the FlowerClient class. Each client is dynamically initialized, trains on its partitioned data, and incorporates Differential Privacy (DP) to protect data by adding noise to updates before they are sent to the server.

###### Server-Side Operations (server.py):

Defines the central server's responsibilities in aggregating model updates from clients using Secure Aggregation, which ensures that sensitive client information remains private. The server also coordinates training sessions and aggregates parameters and evaluation metrics from clients.

###### Federated Learning Orchestration (main.py):

The entry point for running the federated learning simulation. This script sets up the clients, dataset, and training strategy, and coordinates multiple rounds of communication between clients and the server, ensuring that results are logged after each round.

###### Configuration via base.yaml:

The 'conf/base.yaml' file contains essential configurations for the federated learning pipeline. 
It allows you to easily modify important parameters before running the simulation, such as:

```python
num_rounds: 2                  # Number of communication rounds between clients and the server
num_clients: 5                 # Total number of clients
batch_size: 32                 # Batch size for local training
num_classes: 10                # Number of output classes (e.g., for MNIST or CIFAR10)
num_clients_per_round_fit: 5   # Number of clients participating in training per round
num_clients_per_round_eval: 5  # Number of clients participating in evaluation per round
config_fit: 
      lr: 0.001                # Learning rate for local training
      weight_decay: 1e-4       # Weight decay to avoid overfitting
      momentum: 0.9            # Momentum for optimization
      local_epochs: 2          # Number of local training epochs per client
      dp:
      epsilon: 1.0             # Privacy budget for Differential Privacy
      delta: 1e-5              # Privacy parameter controlling the risk of leakage
      noise_scale: 0.01        # Amount of noise added to model updates for DP

```
By adjusting parameters like *num_rounds*, *num_clients*, *batch_size*, and *learning rate*, you can tailor the simulation to your specific requirements. This file is crucial for controlling how the federated learning process operates, making it flexible and adaptable to different experiments.


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

### Google Colab Notebook

Alternatively, you can run this project using the provided Google Colab notebook. It allows you to easily run the entire Federated Learning pipeline in the cloud without the need to set up a local environment.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the [LICENSE](LICENSE) file for more details.
