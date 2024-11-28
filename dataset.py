from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def getDataLoader(config):
    """
    Seleciona o carregador com base no dataset configurado.
    """
    loaders = {
        'mnist': mnist_loaders,
        'cars': cars_loaders
    }
    if config.dataset not in loaders:
        raise ValueError(f"Dataset {config.dataset} não suportado.")
    return loaders[config.dataset](config)

def mnist_loaders(config):
    """
    Carregador do dataset MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
        # Adicione transforms.Normalize se necessário
    ])

    trainLoader = DataLoader(
        MNIST('data', train=True, download=True, transform=transform),
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True
    )

    testLoader = DataLoader(
        MNIST('data', train=False, download=True, transform=transform),
        batch_size=config.test_batch_size,
        shuffle=False,
        pin_memory=True
    )

    return trainLoader, testLoader

def cars_loaders(config):
    """
    Carregador do dataset de carros.
    """
    dataset_path = '/kaggle/input/cars-image-dataset/Cars Dataset'
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Altere o tamanho se necessário
        transforms.ToTensor()
        # Adicione transforms.Normalize se necessário
    ])

    train_path = os.path.join(dataset_path, 'train')  # Subpasta Train
    test_path = os.path.join(dataset_path, 'test')    # Subpasta Test

    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    trainLoader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True)
    testLoader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True)

    return trainLoader, testLoader

# Exemplo de uso
class Config:
    train_batch_size = 64
    test_batch_size = 64
    dataset = 'cars'

config = Config()

trainLoader, testLoader = getDataLoader(config)
