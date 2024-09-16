'''
# client.py
This script defines the FlowerClient class, which is responsible for managing the local training and evaluation on each client. 
In addition, it ensures that each client is correctly initialized, receives updates from the server, and utilizes differential privacy
techniques to protect the data. Additionally, each client is generated dynamically and initialized only once.
'''
import torch
import flwr as fl
from typing import Dict
from flwr.common import NDArrays, Scalar, Context
from model import MNISTNet, CIFAR10Net, train, test
import logging
import numpy as np

# Setting up loggig up important information like initialization, training, and evaluation
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)


initialized_clients = set()               # This set help initializ clients to avoid accidentally re-initializing the same client

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, dataset_choice, client_id, epsilon=0.1, delta=1e-5, noise_scale=0.01) -> None:
        """
        Initializes the client for federated learning with differential privacy settings.
        
        Arguments:
            trainloader: DataLoader for training data.
            valloader: DataLoader for validation data.
            num_classes: Number of output classes (10 for MNIST or CIFAR-10).
            dataset_choice: The dataset being used (MNIST or CIFAR10).
            client_id: Unique ID for each client.
            epsilon: Privacy budget for DP.
            delta: Privacy tolerance for DP.
            noise_scale: Scale of noise added for DP.
        """
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.dataset_choice = dataset_choice
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = noise_scale

        # Choose the appropriate model based on the dataset (MNIST or CIFAR10)
        if self.dataset_choice == "mnist":
            self.model = MNISTNet(num_classes=num_classes)
        else:
            self.model = CIFAR10Net(num_classes=num_classes)

        self.model.to(self.device)                  # Move the model to the appropriate device (GPU or CPU)

        if self.client_id in initialized_clients:   # Check if this client has already been initialized; if yes, skip initialization
            logger.warning(f"[Client {self.client_id}] already initialized, skipping.")
        else:
            initialized_clients.add(self.client_id)
            logger.info(f"[Client {self.client_id}] Initialized with DP settings (ε={self.epsilon}, δ={self.delta}, noise_scale={self.noise_scale})")


    def set_parameters(self, parameters: NDArrays):
        """Sets the model parameters received from the server."""
        
        params_dict = self.model.state_dict()  # Get the model's parameters
        
        for i, key in enumerate(params_dict.keys()):
            params_dict[key] = torch.tensor(parameters[i], dtype=params_dict[key].dtype).to(self.device)
        self.model.load_state_dict(params_dict, strict=True)


    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Gets the model parameters to send back to the server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]  # Convert parameters to NumPy arrays


    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Trains the model using the current parameters and returns the updated parameters."""
        
        self.set_parameters(parameters)         # Set the model parameters received from the server


        lr = config.get('lr', 0.001)            # Set training configurations
        weight_decay = config.get('weight_decay', 1e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = config.get('local_epochs', 1)
        clip_value = config.get('clip_value', 1)

        # Training the model
        train_loss, train_accuracy = train(self.model, self.trainloader, optimizer, epochs, self.device)


        # Apply differential privacy to the model's parameters before sending them to the server
        dp_parameters = self.apply_differential_privacy(self.model.state_dict())


        # Logging that DP has been applied with specific settings
        logger.info(f"[Client {self.client_id}] Applied Differential Privacy (ε={self.epsilon}, δ={self.delta})")


        return [dp_parameters[key].cpu().numpy() for key in dp_parameters], len(self.trainloader), {"loss": train_loss, "accuracy": train_accuracy}


    def apply_differential_privacy(self, state_dict):
        """Adds noise to the model parameters to ensure differential privacy."""
        
        dp_parameters = {}
        for name, param in state_dict.items():
            if param.dtype == torch.long:
                param = param.float()                           # Convert to float for noise addition
            noise = torch.randn_like(param) * self.noise_scale  # Generate noise based on the noise scale
            dp_parameters[name] = param + noise                 # Add noise to the parameters
        return dp_parameters


    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluates the model on the validation data and returns the loss and accuracy."""
        
        self.set_parameters(parameters)                         # Set the model parameters received from the server

        # Evaluate the model on the validation data
        eval_loss, eval_accuracy = test(self.model, self.valloader, self.device)

        logger.info(f"Client {self.client_id} evaluation results: Loss = {eval_loss:.4f}, Accuracy = {eval_accuracy:.4f}")

        return float(eval_loss), len(self.valloader), {"loss": eval_loss, "accuracy": eval_accuracy}



def generate_client_fn(trainloaders, valloaders, num_classes, dataset_choice, epsilon=0.1, delta=1e-5, noise_scale=0.01):
    """
    This function generates a unique initialization function for each client, ensuring that it is dynamically generated.
    
    Arguments:
        trainloaders: List of training DataLoaders
        valloaders: List of validation DataLoaders
        num_classes: Number of output classes for the model
        dataset_choice: The dataset being used ('MNIST' or 'CIFAR10').
        epsilon: Privacy budget for DP.
        delta: Privacy tolerance for DP.
        noise_scale: Scale of noise added for DP.
    
    Returns:
        A function that initializes a unique FlowerClient for each client. 
        It is necessary to start the simulation of federated learning environment using FLower framework
    """


    def client_fn(context: Context):
        cid = int(context.node_id) % len(trainloaders)              # Make sure the client ID is within a valid range
        logger.info(f"Initializing client {cid}") 
        
        return FlowerClient(
                            trainloader=trainloaders[cid],
                            valloader=valloaders[cid],
                            num_classes=num_classes,
                            dataset_choice=dataset_choice,
                            client_id=cid,
                            epsilon=epsilon,
                            delta=delta,
                            noise_scale=noise_scale
                        ).to_client()
    
    return client_fn
