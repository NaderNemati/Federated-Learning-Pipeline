from asyncio.log import logger
from collections import OrderedDict
import torch
import flwr as fl
from typing import Dict, Any
from flwr.common import NDArrays, Scalar, Context
from model import MNISTNet, CIFAR10Net, train, test
import logging  # Add logging

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, dataset_choice, client_id) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.dataset_choice = dataset_choice
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client_id = client_id

        if self.dataset_choice == "mnist":
            self.model = MNISTNet(num_classes=num_classes)
        else:
            self.model = CIFAR10Net(num_classes=num_classes)
        
        self.model.to(self.device)

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for i, (key, param) in enumerate(state_dict.items()):
            if i < len(parameters):  
                param.data = torch.tensor(parameters[i]).data
            else:
                logging.info(f"Skipping parameter {key} due to size mismatch.")
        self.model.load_state_dict(state_dict, strict=False)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 1e-4)

        # Use AdamW optimizer for training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = config.get('local_epochs', 1)

        # Train the model and capture the loss and accuracy
        train_loss, train_accuracy = train(self.model, self.trainloader, optimizer, epochs, self.device)

        # Return the parameters, length of the training data, and the metrics (including loss and accuracy)
        return self.get_parameters({}), len(self.trainloader), {"loss": train_loss, "accuracy": train_accuracy}


    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        client_id = config.get('client_id', 'unknown')

        # Evaluate the model
        eval_loss, eval_accuracy = test(self.model, self.valloader, self.device)

        # Log client-specific evaluation results
        logger.info(f"Client {client_id} evaluation results: Loss = {eval_loss:.4f}, Accuracy = {eval_accuracy:.4f}")

        # Return the loss and accuracy
        return float(eval_loss), len(self.valloader), {"loss": eval_loss, "accuracy": eval_accuracy}


# Function to generate clients
def generate_client_fn(trainloaders, valloaders, num_classes, dataset_choice):
    def client_fn(context: Context):
        cid = int(context.node_id) % len(trainloaders)
        logging.info(f"Initializing client {cid}")
        return FlowerClient(
            trainloader=trainloaders[cid],
            valloader=valloaders[cid],
            num_classes=num_classes,
            dataset_choice=dataset_choice,
            client_id=cid
        ).to_client()
    return client_fn
