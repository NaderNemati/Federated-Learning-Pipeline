'''
main.py
This script is the entry point for this federated learning simulation using the Flower framework.
It sets up the simulation, including the dataset, clients, and training strategy, then runs the federated learning process over multiple rounds. 
Furthermore, it applies secure aggregation to ensure privacy and logs the results after each round.
'''
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
from server import get_on_fit_config, get_evaluate_fn, SecureAggregationFedAvg, fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                                    
os.environ['HYDRA_FULL_ERROR'] = '1'                                        
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                                   


# Configuration for logging and displaying the progress of the FL simulation
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    dataset_choice = input("Choose a dataset for FL simulation (MNIST/CIFAR10): ").strip().lower()          # Prompt the user to select a dataset (MNIST or CIFAR10)

    if dataset_choice not in ["mnist", "cifar10"]:                                                          # Validate the dataset choice and ensure it's either MNIST or CIFAR10
        raise ValueError("Invalid dataset choice. Please choose 'MNIST' or 'CIFAR10'.")

    # Logging the basic configuration parameters
    logger.info(f"num_rounds: {cfg.num_rounds}, num_clients: {cfg.num_clients}, batch_size: {cfg.batch_size}")
    logger.info(f"num_clients_per_round_fit: {cfg.num_clients_per_round_fit}, num_clients_per_round_eval: {cfg.num_clients_per_round_eval}")
    logger.info(f"Chosen dataset: {dataset_choice.upper()}")

    # Preparation of the dataset by splitting it into client-based loaders for training, validation, and testing
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size, dataset_choice)
    logger.info(f"Number of clients: {cfg.num_clients}, Client 0 dataset size: {len(trainloaders[0].dataset)}")

    # Client function generation to initialize clients using the relevant data and privacy settings
    client_fn = generate_client_fn(
                                trainloaders, 
                                validationloaders, 
                                cfg.num_classes, 
                                dataset_choice, 
                                epsilon=cfg.dp.epsilon, 
                                delta=cfg.dp.delta, 
                                noise_scale=cfg.dp.noise_scale
                                )


    # Dederated learning strategy definition with secure aggregation
    strategy = SecureAggregationFedAvg(
                fraction_fit=1.0,                                                               # All clients participate in training each round
                min_fit_clients=cfg.num_clients,                                                # Minimum number of clients needed for training
                fraction_evaluate=1.0,                                                          # All clients participate in evaluation each round
                min_evaluate_clients=cfg.num_clients,                                           # Minimum number of clients needed for evaluation
                min_available_clients=cfg.num_clients,                                          # Minimum clients that need to be available
                on_fit_config_fn=get_on_fit_config(cfg),                                        # Configuration settings for each training round
                evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, dataset_choice),       # Function to evaluate the global model
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,                          # Aggregation of training metrics
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,                # Aggregation of evaluation metrics
    )


    # Start point of the the Flower simulation with the defined client function, strategy, and number of rounds
    logger.info("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),                               # Number of rounds for the simulation
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.0}                                       # Resource usage handling per client
    )


    # Saving the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'
    with open(str(results_path), 'wb') as h:
        pickle.dump({'history': history}, h, protocol=pickle.HIGHEST_PROTOCOL)


    # Logging a summary of the completed simulation
    logger.info(f"Run finished {cfg.num_rounds} round(s)")


    # Displaying distributed training loss for each round
    if hasattr(history, 'losses_distributed'):
        logger.info("\nLoss (Distributed Training):")
        for i, loss in enumerate(history.losses_distributed, 1):
            logger.info(f"    Round {i}: loss = {loss}")
        if len(history.losses_distributed) < cfg.num_rounds:
            missing_rounds = set(range(1, cfg.num_rounds + 1)) - set(range(1, len(history.losses_distributed) + 1))
            logger.info(f"    Missing loss data for rounds: {missing_rounds}")


    # Displaying centralized evaluation loss for each round
    if hasattr(history, 'losses_centralized'):
        logger.info("\nLoss (Centralized Evaluation):")
        for i, loss in enumerate(history.losses_centralized, 0):
            logger.info(f"    Round {i}: {loss[0]:.2f}")


    # Displaying distributed training accuracy for each round
    if hasattr(history, 'metrics_distributed'):
        logger.info("\nAccuracy (Distributed Training):")
        for i, (round_num, acc) in enumerate(history.metrics_distributed.get("accuracy", []), 1):
            logger.info(f"    Round {i}: {acc * 100:.2f}%")
        if len(history.metrics_distributed.get("accuracy", [])) < cfg.num_rounds:
            missing_rounds = set(range(1, cfg.num_rounds + 1)) - set(range(1, len(history.metrics_distributed.get("accuracy", [])) + 1))
            logger.info(f"    Missing accuracy data for rounds: {missing_rounds}")


    # Logging that secure aggregation has been applied in each round
    logger.info("Secure Aggregation applied in each round")


    # Final logging statement indicating that the simulation has completed
    logger.info("Flower simulation finished")




if __name__ == "__main__":
    main()
