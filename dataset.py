from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def getDataLoader(config):
    loaders = {
        'mnist': mnist_loaders,
        'imagefolder': imagefolder_loaders,
        'cifar10': cifar10_loaders,
        'vehicle': vehicle_loaders,
        'lisa': lisa_loaders,
        'bstl': bstl_loaders
    }[config.dataset]
    return loaders(config)

def mnist_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        #transforms.Normalize(mean, std)
    ])
    
    trainLoader = DataLoader(MNIST('data', train=True, download=True, transform=transform),
                             batch_size=config.train_batch_size, shuffle=True, pin_memory=True)

    testLoader = DataLoader(MNIST('data', train=False, download=True, transform=transform),
                            batch_size=config.test_batch_size, shuffle=False, pin_memory=True)
    
    return trainLoader, testLoader

def cifar10_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])
    
    trainLoader = DataLoader(CIFAR10('data', train=True, download=True, transform=transform),
                             batch_size=config.train_batch_size, shuffle=True, pin_memory=True)

    testLoader = DataLoader(CIFAR10('data', train=False, download=True, transform=transform),
                            batch_size=config.test_batch_size, shuffle=False, pin_memory=True)

    return trainLoader, testLoader

def imagefolder_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])
    
    data_dir = '/kaggle/input/cars-image-dataset/Cars Dataset'
    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"

    train_data = ImageFolder(root=train_dir, transform=transform)
    test_data = ImageFolder(root=test_dir, transform=transform)

    trainLoader = DataLoader(train_data, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True)
    testLoader = DataLoader(test_data, batch_size=config.test_batch_size,
                            shuffle=False, pin_memory=True)

    return trainLoader, testLoader

import torch

def lisa_loaders(config):
    path = '/kaggle/input/cropped-lisa-traffic-light-dataset'
    train_dir = f"{path}/cropped_lisa_1/train_1"
    val_dir = f"{path}/cropped_lisa_1/val_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if config.normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)
    
    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def vehicle_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],  # ImageNet mean
        #    std=[0.229, 0.224, 0.225]    # ImageNet std
        #)
    ])
    test_split = 0.2
    data_dir = '/kaggle/input/vehicle-detection-image-set/data'

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    trainLoader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
    
    return trainLoader, testLoader

def bstl_loaders(config):
    train_dir = "/kaggle/input/bstl-dataset/train"
    test_dir = "/kaggle/input/bstl-dataset/test"

    transform_list = [transforms.Resize((32, dim)), transforms.ToTensor()]

    if config.normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    print(f"Número de imagens em train: {len(train_dataset)}")
    print(f"Número de imagens em test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader
