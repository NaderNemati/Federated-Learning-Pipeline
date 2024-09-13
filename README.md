<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f4f4f4;
            padding: 20px;
        }

        h1 {
            color: #0073e6;
        }

        h2, h3 {
            color: #005bb5;
        }

        code {
            background-color: #f9f9f9;
            padding: 2px 6px;
            font-size: 14px;
            border-radius: 5px;
        }

        table {
            border-collapse: collapse;
            margin: 20px 0;
            width: 100%;
        }

        table, th, td {
            border: 1px solid #ddd;
        }

        th, td {
            padding: 8px;
            text-align: left;
        }

        .folder-structure {
            background-color: #eee;
            border: 1px solid #ccc;
            padding: 15px;
            font-family: Consolas, monospace;
        }

        .folder-structure pre {
            white-space: pre-line;
        }

        .important-note {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 10px;
            margin: 20px 0;
        }

        a {
            color: #0073e6;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        .content-block {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
    </style>
</head>

<body>
    <h1>Federated Learning Pipeline</h1>
    <p>This project demonstrates a Federated Learning pipeline using Flower and PyTorch to train a CNN model for image classification on the MNIST dataset.</p>

    <div class="content-block">
        <h2>Project Structure</h2>
        <div align="center">
            <img src="https://raw.githubusercontent.com/NaderNemati/Federated-Learning-Pipeline/main/federated-learning-pipeline.png" alt="Project Structure" style="width: 50%; height: auto;">
        </div>
        <div class="folder-structure">
            <pre>Federated-Learning-Pipeline/
├── client.py           # Client-side code for federated learning
├── dataset.py          # Dataset loading and partitioning
├── main.py             # Main entry point for running the federated learning simulation
├── model.py            # Model definition (CNN for MNIST)
├── server.py           # Server-side federated learning configuration
├── requirements.txt    # Dependencies to run the code
├── README.md           # Documentation for the project
├── LICENSE             # License file
└── conf/               # Configuration files for the simulation (base.yaml)
            </pre>
        </div>
    </div>

    <div class="content-block">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#usage">Usage</a></li>
            <li><a href="#project-structure">Project Structure</a></li>
            <li><a href="#contributing">Contributing</a></li>
            <li><a href="#license">License</a></li>
        </ul>
    </div>

    <div class="content-block" id="overview">
        <h2>Overview</h2>
        <p>This repository implements a Federated Learning pipeline using the Flower framework, where multiple clients collaboratively train a model on the MNIST dataset for image classification. The pipeline includes:</p>
        <ul>
            <li>A <code>client.py</code> file, which defines the Flower client to handle local training on MNIST data.</li>
            <li>A <code>server.py</code> file, defining server-side configurations for federated learning rounds.</li>
            <li>A <code>dataset.py</code> file for partitioning the MNIST dataset among clients.</li>
            <li>A <code>model.py</code> file, defining a simple Convolutional Neural Network (CNN) model for classification.</li>
            <li>A <code>main.py</code> file to orchestrate the federated learning simulation using Flower.</li>
        </ul>
    </div>

    <div class="content-block" id="installation">
        <h2>Installation</h2>

        <h3>Prerequisites</h3>
        <ul>
            <li>Python 3.9 or later</li>
            <li><a href="https://flower.dev/" target="_blank">Flower</a></li>
            <li><a href="https://pytorch.org/" target="_blank">PyTorch</a></li>
        </ul>

        <h3>Install dependencies</h3>
        <p>To install the required dependencies, run the following command:</p>
        <pre><code>pip install -r requirements.txt</code></pre>
    </div>

    <div class="content-block" id="usage">
        <h2>Usage</h2>

        <h3>Running the Federated Learning Simulation</h3>
        <p>Clone the repository:</p>
        <pre><code>git clone https://github.com/NaderNemati/Federated-Learning-Pipeline.git
cd Federated-Learning-Pipeline</code></pre>

        <h3>Install dependencies:</h3>
        <p>Make sure the virtual environment is activated if you use one, and run:</p>
        <pre><code>pip install -r requirements.txt</code></pre>

        <h3>Run the simulation:</h3>
        <p>To start the federated learning simulation, run the <code>main.py</code> script:</p>
        <pre><code>python main.py</code></pre>

        <p>This command orchestrates the federated learning process by:</p>
        <ul>
            <li>Loading and partitioning the MNIST dataset using <code>dataset.py</code></li>
            <li>Running the federated simulation with client-side logic in <code>client.py</code> and server-side logic in <code>server.py</code></li>
            <li>Training a CNN model defined in <code>model.py</code> on the partitioned dataset.</li>
        </ul>
    </div>

    <div class="content-block">
        <h2>Configuration</h2>
        <p>You can modify the federated learning simulation parameters by editing the <code>conf/base.yaml</code> file.</p>
        <pre><code>num_rounds: 5
num_clients: 5
batch_size: 20
num_classes: 10
num_clients_per_round_fit: 3
num_clients_per_round_eval: 2
config_fit: 
  lr: 0.01
  momentum: 0.9
  local_epochs: 1</code></pre>
    </div>

    <div class="content-block" id="license">
        <h2>License</h2>
        <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    </div>
</body>

</html>
