'''
dataset.py:
This script takes care of loading datasets and splitting them up for our federated learning setup.
The goal here is to support both MNIST and CIFAR10, apply some useful transformations (like normalization), 
and then split the data among multiple clients. This is crucial because, in federated learning, 
clients train models on their own data without sharing it directly, so data preparation is essential.
'''
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop



def get_dataset(dataset_choice: str, data_path: str = './data'):
    """
    Loads the dataset the user selects (either MNIST or CIFAR10) with proper transformations.
    
    Arguments:
        dataset_choice (str): The dataset to load ('mnist' or 'cifar10').
        data_path (str): Where to store the dataset.
        
    Returns:
        Tuple: A tuple containing the training and test datasets.
    
    Raises:
        ValueError: If the dataset_choice isn't 'mnist' or 'cifar10'.
    
    Using this function, we can preprocess MNIST and CIFAR10 in different ways. For example, MNIST is grayscale, so it gets simple normalization, 
    while CIFAR10 gets a bit more fancy treatment with things like random horizontal flips and cropping to make the model more robust.
    """
    
    if dataset_choice == 'mnist':
        # MNIST is pretty straightforward. We convert the images to tensors and normalize them.
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST(data_path, train=True, download=True, transform=transform)
        testset = MNIST(data_path, train=False, download=True, transform=transform)
    
    elif dataset_choice == 'cifar10':
        # Training CIFAR10 includes some extra augmentations to improve generalization and proficiency
        transform_train = Compose([
            RandomHorizontalFlip(),                             # Randomly flip the image horizontally
            RandomCrop(32, padding=4),                          # Crop and pad to simulate data variability
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))         # Normalization of RGB channels
        ])
        
        # There is no need to augment the test set, and normalization is the only processing step.
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = CIFAR10(data_path, train=True, download=True, transform=transform_train)
        testset = CIFAR10(data_path, train=False, download=True, transform=transform_test)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_choice}. Please choose either 'mnist' or 'cifar10'.")
    
    return trainset, testset



def prepare_dataset(num_partitions: int, batch_size: int, dataset_choice: str, val_ratio: float = 0.1):
    """
    Prepares the dataset for federated learning by splitting it into multiple partitions (clients).
    
    Arguments:
        num_partitions (int): Number of clients (partitions) to split the training data into.
        batch_size (int): Batch size to use for the data loaders.
        dataset_choice (str): The dataset choice ('mnist' or 'cifar10').
        val_ratio (float): Each client should use 10% of data for validation
        
    Returns:
        Tuple: A tuple containing data loaders for training, validation, and testing.
    
        Data is divided between clients using the IID method (Independent and Identically Distributed). 
        Due to the fact that our dataset comes from a larger pool with the same distribution (such as MNIST or CIFAR10), 
        we assume that the data drawn from each client is also drawn from the same distribution. Based on the law of large numbers, 
        each client's data represents the overall dataset fairly. Therefore, the IID method must be used to divide the data in order to maintain 
        this consistency and make sure that all clients are trained on the same global data.
    """
    
    # Loading the dataset based on what the user chose (either MNIST or CIFAR10).
    trainset, testset = get_dataset(dataset_choice)
    
    # Spliting the training set into 'num_partitions' (one for each client).
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions                   # Ensure that all clients get the same amount of data.
    
    # Randomly splitting the training data among the clients.
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))


    trainloaders = []
    valloaders = []

    # Splitting related part of the data further into training and validation sets for each client.
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total) 
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        # Appending data loaders for both training and validation sets to their respective lists.
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # Creation of the test loader for the test set (which is shared by all clients).
    # The test set is kept in its original size and used after aggregation to evaluate the global model.
    testloaders = DataLoader(testset, batch_size=128)
    
    return trainloaders, valloaders, testloaders
