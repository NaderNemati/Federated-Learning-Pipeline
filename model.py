'''
model.py
The script defines the models we use for the MNIST and CIFAR10 datasets. 
The appropriate model architecture is dynamically selected and initialized based on the dataset being used. 
As part of our training and testing functions, we also handle optimization, loss calculation, and evaluation. 
We can easily switch between datasets and models with this code because it is designed to be flexible.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST Model Definition
class MNISTNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(MNISTNet, self).__init__()
                                                                
        self.conv1 = nn.Conv2d(1, 32, 5)       # Initial convolutional layers based on grayscale MNIST images (1 channel), 32 filters, 5x5 kernels
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(32, 64, 5)  
        self.fc1 = nn.Linear(64 * 4 * 4, 120)  # Fully connected layers for classification
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, num_classes)  # Output layer with 10 neurons, number of classes for MNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)              # Flatten the output from the conv layers to feed into fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                         # No activation on the final layer (since we use softmax in the loss function)
        return x

# CIFAR10 Model Definition (CIFAR10 images are RGB, so we have 3 input channels. The network here is deeper than the MNIST model.)
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 64 filters, 3x3 kernel size, padding to keep image size the same
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(64)         # Batch normalization helps with training stability and convergence
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)          # Fully connected layers for classification
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)                  # Dropout to prevent overfitting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                              # No activation on the final layer (since we use softmax in the loss function)
        return x


# Function to get the appropriate model based on dataset choice
def get_model(dataset_choice: str, num_classes: int):
    """
    Returns the model corresponding to the dataset being used (MNIST or CIFAR10).
    
    Arguments:
        dataset_choice (str): The dataset we're working with ('mnist' or 'cifar10').
        num_classes (int): The number of output classes (10 for both MNIST and CIFAR10).
    
    Returns:
        torch.nn.Module: The appropriate model (either MNISTNet or CIFAR10Net).
    """
    if dataset_choice.lower() == "mnist":
        return MNISTNet(num_classes)
    elif dataset_choice.lower() == "cifar10":
        return CIFAR10Net(num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_choice}. Please choose either 'mnist' or 'cifar10'.")

# Training function
def train(net, trainloader, optimizer, epochs, device: str, clip_value=None):
    """
    Trains the given network using the specified training data loader and optimizer.
    
    Arguments:
        net: The neural network model to train.
        trainloader: Providing batches of training data.
        optimizer: The optimizer for training (AdamW is used in this project).
        epochs: The number of epochs to train for.
        device (str): The device to run the training on ('cpu' or 'gpu').
        clip_value: The value for gradient clipping to prevent exploding gradients. This is optional
    
    Returns:
        avg_loss: The average loss over the training set.
        accuracy: The accuracy of the model on the training data.
    """
    criterion = torch.nn.CrossEntropyLoss()             # Standard loss function for classification
    net.train() 
    net.to(device)

    correct = 0
    total = 0
    total_loss = 0.0

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Move data to the selected device
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # For CIFAR-10 image classification, gradient clipping is crucial as it prevents exploding gradients, stabilizes training, and ensures controlled weight updates.
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(trainloader)
    
    return avg_loss, accuracy

# Testing function
def test(net, testloader, device: str):
    """
    Evaluates the model's performance on the test set.
    
    Args:
        net: The trained neural network model to evaluate.
        testloader: DataLoader providing batches of test data.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
    
    Returns:
        loss: The total loss on the test set.
        accuracy: The accuracy of the model on the test data.
    """
    criterion = torch.nn.CrossEntropyLoss()                     # Loss function for classification
    correct = 0
    loss = 0.0
    net.eval()
    net.to(device)

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
