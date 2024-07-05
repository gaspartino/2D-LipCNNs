from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
#from torchvision.datasets import VisionDataset
#from torchvision.datasets.folder import default_loader
#from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
#from utils import load_obj

def getDataLoader(config):
    loaders = {
        'mnist': mnist_loaders
    }[config.dataset]
    return loaders(config)

def mnist_loaders(config):

    trainLoader = DataLoader(MNIST('data',train=True,download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32, 32))
                        #transforms.Normalize(mean, std)
                        ])),
                    batch_size=config.train_batch_size,
                    shuffle=True, pin_memory=True)

    testLoader = DataLoader(MNIST('data',
                    train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32, 32))
                        #transforms.Normalize(mean, std)
                    ])),
                batch_size=config.test_batch_size,
                shuffle=False, pin_memory=True)
    
    return trainLoader, testLoader
    

