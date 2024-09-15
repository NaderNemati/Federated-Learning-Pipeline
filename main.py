import logging
import pickle
from pathlib import Path
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress unnecessary logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization logs

# Aggregate functions for metrics
def fit_metrics_aggregation_fn(metrics_list):
    accuracies = [metrics.get("accuracy", 0) for _, metrics in metrics_list]
    losses = [metrics.get("loss", 0) for _, metrics in metrics_list]
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)
    return {"accuracy": avg_accuracy, "loss": avg_loss}

def evaluate_metrics_aggregation_fn(metrics_list):
    accuracies = [metrics.get("accuracy", 0) for _, metrics in metrics_list]
    losses = [metrics.get("loss", 0) for _, metrics in metrics_list]
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)
    return {"accuracy": avg_accuracy, "loss": avg_loss}

# Configuring logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    dataset_choice = input("Choose a dataset for FL simulation (MNIST/CIFAR10): ").strip().lower()

    if dataset_choice not in ["mnist", "cifar10"]:
        raise ValueError("Invalid dataset choice. Please choose 'MNIST' or 'CIFAR10'.")

    logger.info(f"num_rounds: {cfg.num_rounds}, num_clients: {cfg.num_clients}, batch_size: {cfg.batch_size}")
    logger.info(f"num_clients_per_round_fit: {cfg.num_clients_per_round_fit}, num_clients_per_round_eval: {cfg.num_clients_per_round_eval}")
    logger.info(f"Chosen dataset: {dataset_choice.upper()}")

    # Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, dataset_choice)
    logger.info(f"Number of clients: {cfg.num_clients}, Client 0 dataset size: {len(trainloaders[0].dataset)}")

    # Generate client function
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes, dataset_choice)

    # Strategy definition
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=cfg.num_clients,
        fraction_evaluate=1.0,
        min_evaluate_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, dataset_choice),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    logger.info("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.0}
    )

    # Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'
    with open(str(results_path), 'wb') as h:
        pickle.dump({'history': history}, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Updated summary and formatted log output
    logger.info(f"Run finished {cfg.num_rounds} round(s)")

    # Display loss (distributed) history
    if hasattr(history, 'losses_distributed'):
        logger.info("\nLoss (Distributed Training):")
        for i, loss in enumerate(history.losses_distributed, 1):
            logger.info(f"    Round {i}: loss = {loss}")

    # Display loss (centralized) history
    if hasattr(history, 'losses_centralized'):
        logger.info("\nLoss (Centralized Evaluation):")
        for i, loss in enumerate(history.losses_centralized, 0):
            logger.info(f"    Round {i}: {loss[0]:.2f}")

    # Display accuracy (distributed) history
    if hasattr(history, 'metrics_distributed'):
        logger.info("\nAccuracy (Distributed Training):")
        for i, (round_num, acc) in enumerate(history.metrics_distributed.get("accuracy", []), 1):
            logger.info(f"    Round {i}: {acc * 100:.2f}%")

    # Display accuracy (centralized) history
    if hasattr(history, 'metrics_centralized'):
        logger.info("\nAccuracy (Centralized Evaluation):")
        for i, (round_num, acc) in enumerate(history.metrics_centralized.get("accuracy", []), 0):
            logger.info(f"    Round {i}: {acc * 100:.2f}%")

    logger.info("Flower simulation finished")

if __name__ == "__main__":
    main()
