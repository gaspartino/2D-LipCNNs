import torch
import torch.linalg as la
import numpy as np
import torchattacks
from torchvision.transforms import Normalize
from model import getModel
from dataset import getDataLoader
from utils import *
import matplotlib.pyplot as plt

def add_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[f"module.{k}"] = v
    return new_state_dict

def plot_accuracies(epsilons, accuracies):

    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, marker='o')
    plt.title("Model Accuracy vs. FGSM Attack Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def accuracy_AA(model, dataset_loader, num_classes, eps, device):
    attack = torchattacks.AutoAttack(model, norm='Linf', eps=eps, version='standard', n_classes=num_classes)
    total_correct = 0
    total_samples = 0

    num_batches = len(dataset_loader)

    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)

        with torch.no_grad():
            predictions = model(x_adv)
            predicted_class = predictions.argmax(dim=1)

        correct = (predicted_class == y).sum().item()
        total_correct += correct
        total_samples += y.size(0)

        # Mostrar feedback batch a batch
        batch_acc = correct / y.size(0)
        total_acc = total_correct / total_samples
        print(f"[{i+1}/{num_batches}] Batch acc: {batch_acc:.4f} | Total acc até agora: {total_acc:.4f}")

    return total_correct / total_samples

import torch
import torchattacks

def accuracy_FGSM(model, dataset_loader, eps, device):
    attack = torchattacks.FGSM(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)

        with torch.no_grad():
            predictions = model(x_adv)
            predicted_class = predictions.argmax(dim=1)

        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples

def accuracy_PGD(model, dataset_loader, eps, device):
    attack = torchattacks.PGD(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)

        with torch.no_grad():
            predictions = model(x_adv)
            predicted_class = predictions.argmax(dim=1)

        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples

def accuracy_MIM(model, dataset_loader, eps, device):
    attack = torchattacks.MIFGSM(model, eps=eps)
    total_correct = 0
    total_samples = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)

        with torch.no_grad():
            predictions = model(x_adv)
            predicted_class = predictions.argmax(dim=1)

        total_correct += (predicted_class == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples

def accuracy_clean(model, dataloader, device):
    """Acurácia em dados limpos (sem ataques)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            # outputs pode ser logits; usamos argmax para predizer
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def PGDL2_attack(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    model = getModel(config).to(device)
    model_state = torch.load(f"{config.train_dir}/model.ckpt")

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    x = torch.rand((1, config.in_channels, config.img_size, config.img_size)).to(device)
    model(x)

    _, testLoader = getDataLoader(config)

    epsilons = [0.01, 0.02, 8/255, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    accuracies = []

    results = {}

    clean_acc = accuracy_clean(model, testLoader, device)
    print(f"Accuracy on clean data: {round(clean_acc * 100, 2)}%")
    results['clean'] = clean_acc

    for epsilon in epsilons:
        accuracy = accuracy_FGSM(model, testLoader, epsilon, device)
        print(f"Accuracy on FGSM (ε={round(epsilon,2)}): {round(accuracy * 100, 2)}%")
        accuracies.append(accuracy)
    print("")
    for epsilon in epsilons:
        accuracy = accuracy_MIM(model, testLoader, epsilon, device)
        print(f"Accuracy on MIM (ε={round(epsilon,2)}): {round(accuracy * 100, 2)}%")
        accuracies.append(accuracy) 
    print("")
    for epsilon in epsilons:
        accuracy = accuracy_PGD(model, testLoader, epsilon, device)
        print(f"Accuracy on PGD (ε={round(epsilon,2)}): {round(accuracy * 100, 2)}%")
        accuracies.append(accuracy)
    print("")  

    for epsilon in epsilons:
        accuracy = accuracy_AA(model, testLoader, 7, epsilon, device)
        print(f"Accuracy on AutoAttack (ε={round(epsilon,2)}): {round(accuracy * 100, 2)}%")
        accuracies.append(accuracy)
        
    return accuracies
