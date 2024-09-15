import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomCrop


def get_dataset(dataset_choice: str, data_path: str = './data'):
    """Loads the selected dataset (MNIST or CIFAR10)"""
    if dataset_choice == 'mnist':
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST(data_path, train=True, download=True, transform=transform)
        testset = MNIST(data_path, train=False, download=True, transform=transform)
    
    elif dataset_choice == 'cifar10':
        # Adding data augmentation techniques for CIFAR10
        transform_train = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = CIFAR10(data_path, train=True, download=True, transform=transform_train)
        testset = CIFAR10(data_path, train=False, download=True, transform=transform_test)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_choice}")
    
    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, dataset_choice: str, val_ratio: float = 0.1):
    """Prepares and partitions the dataset based on the user's choice."""
    trainset, testset = get_dataset(dataset_choice)
    
    # Split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))

    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloaders = DataLoader(testset, batch_size=128)
    
    return trainloaders, valloaders, testloaders
