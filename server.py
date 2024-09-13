'''
from collections import OrderedDict
from omegaconf import DictConfig
import torch
from model import Net, test

from typing import Dict, Any
from flwr.common import NDArrays, Scalar

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):


        #if server_round > 50:
        #    lr = config.lr / 10
        return {'lr': config.lr, 'momentum': config.momentum,
                'local_epochs': config.local_epochss}
    
    return fit_config_fn



def get_evaluate_fn(num_classes: int, testloader):



    #def evaluate_fn(server_round: int, parameters, config):
    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.model.load_state_dict(state_dict, strict=True)


        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn()
'''
from collections import OrderedDict
from omegaconf import DictConfig
import torch
from model import Net, test

from typing import Dict, Any
from flwr.common import NDArrays, Scalar

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):


        #if server_round > 50:
        #    lr = config.lr / 10
        return {'lr': config.lr, 'momentum': config.momentum,
                'local_epochs': config.local_epochss}
    
    return fit_config_fn

def get_evaluate_fn(num_classes, testloader):
    """Return an evaluation function for the server to use."""

    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluation logic during federated learning rounds."""
        # Load the model with the given parameters
        model = Net(num_classes)
        state_dict = zip(model.state_dict().keys(), parameters)
        model.load_state_dict({k: torch.tensor(v) for k, v in state_dict}, strict=True)

        # Evaluate the model on the test dataset
        loss, accuracy = test(model, testloader, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        # Return the loss and accuracy
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate_fn  # Return the function itself, not the result of calling it
