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

## Project Structure (Detailed)

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
