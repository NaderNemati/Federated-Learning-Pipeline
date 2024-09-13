# Federated-Learning-Pipeline

This repository implements a Federated Learning pipeline using Flower and PyTorch, enabling collaborative training of a model on the MNIST dataset for image classification across multiple clients without sharing raw data.


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

## Federated Learning Pipeline on MNIST Dataset

This repository implements a Federated Learning pipeline using the Flower framework. Multiple clients collaboratively train a model on the MNIST dataset for image classification, preserving privacy by training on their local data partitions without sharing raw data.  

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
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
num_rounds: 5
num_clients: 5
batch_size: 20
num_classes: 10
num_clients_per_round_fit: 3
num_clients_per_round_eval: 2
config_fit: 
  lr: 0.01
  momentum: 0.9
  local_epochs: 1
```

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the [LICENSE](LICENSE) file for more details.
