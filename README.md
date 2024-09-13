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

## Project Structure (Fenced Code Block):
