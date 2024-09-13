from collections import OrderedDict
import torch
import flwr as fl
from typing import Dict, Any
from flwr.common import NDArrays, Scalar, Context
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Copy parameters sent by the server into the client's local model
        self.set_parameters(parameters)

        lr = config.get('lr', 0.001)
        momentum = config.get('momentum', 0.9)
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        epochs = config.get('local_epochs', 1)
        # Perform local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        print("Evaluating on validation set...")
        loss, accuracy = test(self.model, self.valloader, self.device)
        print(f"Evaluation results: Loss = {loss}, Accuracy = {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": accuracy}



def generate_client_fn(trainloaders, valloaders, num_classes):
    def client_fn(context):
        node_id = context.node_id
        print(f"Using node_id: {node_id} as client identifier")

        # Use the node_id to select the appropriate client
        cid = int(node_id) % len(trainloaders)

        # Return a Client instance by calling `to_client()`
        return FlowerClient(
            trainloader=trainloaders[cid],
            valloader=valloaders[cid],
            num_classes=num_classes
        ).to_client()  # Ensure the client is returned as `Client`, not `NumPyClient`

    return client_fn

