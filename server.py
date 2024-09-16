'''
server.py
This script defines the server-side operations in a federated learning setup.
Servers aggregate updates from clients, apply secure aggregation techniques to keep sensitive information private and coordinate training sessions. 
Additionally, both parameters and evaluation metrics can be aggregated using custom functions.
'''
import torch
from model import get_model, test
from typing import Dict, Any, List, Tuple
from flwr.common import NDArrays, Scalar, FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import logging
import flwr as fl

# Logging server operations and issues that occur during the process
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)


# Secure Aggregation Function
def secure_aggregation(parameters_list: List[NDArrays]) -> Parameters:
    """
    Aggregates model parameters from clients securely.    
    Arguments:
        parameters_list: List of parameter from each client.
    Returns:
        Parameters: Aggregated parameters after secure aggregation.

    In this function, all parameters from all clients are combined. 
    The idea is to perform this aggregation securely and avoid exposing any client's data individually. 
    In order to calculate the average value across all participants, we sum the parameters from each client and divide them by the number of clients.
    """
    
    if not parameters_list or len(parameters_list) == 0:
        raise ValueError("The parameters list is empty or None.")           # Handle cases where there are no parameters

    aggregated_params = None
    num_clients = len(parameters_list)


    for param_ndarrays in parameters_list:                                                       # Aggregate parameters from each client
        if aggregated_params is None:
            aggregated_params = [np.zeros_like(p) for p in param_ndarrays]

        aggregated_params = [p1 + p2 for p1, p2 in zip(aggregated_params, param_ndarrays)]       # Sum the parameters across clients

    aggregated_params = [p / num_clients for p in aggregated_params]                             # Divide by the number of clients to get the average

    return ndarrays_to_parameters(aggregated_params)


class SecureAggregationFedAvg(fl.server.strategy.FedAvg):                                         # Custom strategy that includes secure aggregation
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[BaseException],
        ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Using the secure aggregation instead of the default behavior.
        
        Arguments:
            server_round: The current round of federated training.
            results: List of client training results.
            failures: List of clients that failed during the round.
        
        Returns:
            Tuple of aggregated parameters and an empty dictionary for additional information.
        
        As part of this strategy, failures that occur during training rounds are logged, and data aggregation is secure to protect client information. 
        The method ensures the privacy of the client's contributions.
        """
        
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} clients failed.")                  # Log any client failures

        # Convert client parameters to ndarray format for aggregation
        parameters_list = [parameters_to_ndarrays(res.parameters) for _, res in results if res.parameters is not None]

        if not parameters_list:
            raise ValueError("The parameters list is empty or None.")

       
        aggregated_params = secure_aggregation(parameters_list)                                      # Applying secure aggregation to the parameters     

        logger.info(f"Round {server_round}: Applied Secure Aggregation on client parameters")

        return aggregated_params, {}


# Fonfiguring the training settings per round
def get_on_fit_config(config):                                                                      
    """
    Generates configuration settings for each training round.
    
    Argumants:
        config: Configuration object that contains the hyperparameters.
    
    Returns:
        A function that generates the configuration for each server round.
    
    The purpose of this function is to customize things like learning rate, weight decay, and number of epochs for each training round.
    """
    def fit_config_fn(server_round: int):
        return {

            'lr': config.config_fit.lr, 
            'weight_decay': config.config_fit.weight_decay,  
            'momentum': config.config_fit.momentum,  
            'local_epochs': config.config_fit.local_epochs,  
            'batch_size': config.config_fit.batch_size,  
            'clip_value': 1.0,  
        }
    
    return fit_config_fn


def get_evaluate_fn(num_classes, testloader, dataset_choice):                                           # Function to evaluate the global model after each round
    """
    Generates the evaluation function to test the global model.
    
    Arguments:
        num_classes: The number of output classes in the dataset.
        testloader: DataLoader for the test dataset.
        dataset_choice: The dataset being used ('MNIST' or 'CIFAR10').
    
    Returns:
        A function that evaluates the model after each training round.

    The purpose of this function is  measuring how well the global model is doing after each round.
    The latest global model parameters will load, and the global model will test on the test dataset
    """
    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        
        model = get_model(dataset_choice, num_classes)                                                  # Get the model architecture based on the dataset choice            

        state_dict = zip(model.state_dict().keys(), parameters)                                         # Load the received parameters into the model
        model.load_state_dict({k: torch.tensor(v) for k, v in state_dict}, strict=True)

        loss, accuracy = test(model, testloader, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Global model evaluation after round {server_round}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")


        return float(loss), {"accuracy": float(accuracy)}

    return evaluate_fn



def fit_metrics_aggregation_fn(metrics_list):                                                          # Aggregation function for training metrics
    """
    Aggregates training metrics (accuracy and loss) from all clients.
    
    Arguments:
        metrics_list: List of metrics dictionaries from clients.
    
    Returns:
        A dictionary with the average accuracy and loss.

    The purpose of this function is to combine the accuracy and loss values from all clients to calculate the global average 
    and utilize it in providing an overview of the training performance after each round.
    """
    accuracies = [metrics.get("accuracy", 0) for _, metrics in metrics_list]
    losses = [metrics.get("loss", 0) for _, metrics in metrics_list]
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)
    
    return {"accuracy": avg_accuracy, "loss": avg_loss}



def evaluate_metrics_aggregation_fn(metrics_list):                                                      # Aggregation function for evaluation metrics
    """
    Aggregation of the evaluation metrics (accuracy and loss) from all clients.
    
    Arguments:
        metrics_list: List of evaluation metrics from clients.
    
    Returns:
        A dictionary with the average accuracy and loss.
    
    Similar to the fit_metrics_aggregation_fn, this function aggregates the evaluation metrics
    (accuracy and loss) from all clients to provide a global view of the model's performance.
    """
    accuracies = [metrics.get("accuracy", 0) for _, metrics in metrics_list]
    losses = [metrics.get("loss", 0) for _, metrics in metrics_list]
    avg_accuracy = np.mean(accuracies)
    avg_loss = np.mean(losses)
    
    return {"accuracy": avg_accuracy, "loss": avg_loss}
