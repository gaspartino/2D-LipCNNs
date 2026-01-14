import torch
import numpy as np
import torchattacks
import torch.linalg as la
import matplotlib.pyplot as plt
from utils import *
from model import getModel
from dataset import getDataLoader
from torchvision.transforms import Normalize
from sklearn.metrics import precision_score, recall_score, f1_score

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

def evaluate_model(model, dataset_loader, attack=None, attack_name="Clean", device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true_all, y_pred_all = [], []

    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)

        if attack is not None:
            x = attack(x, y)

        with torch.no_grad():
            outputs = model(x)
            predicted_class = outputs.argmax(dim=1)

        y_true_all.extend(y.cpu().numpy())
        y_pred_all.extend(predicted_class.cpu().numpy())

    acc = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"\n=== {attack_name} RESULTS ===")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%\n")

    return acc, precision, recall, f1

def accuracy_clean(model, dataset_loader, device):
    return evaluate_model(model, dataset_loader, None, "Clean", device)

def accuracy_FGSM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.FGSM(model, eps=eps)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"FGSM (ε={eps})", device)

def accuracy_PGD(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.PGD(model, eps=eps)   
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"PGD (ε={eps})", device)

def accuracy_MIM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.MIFGSM(model, eps=eps) 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"MIM (ε={eps})", device)
    
def accuracy_AutoAttack(model, dataset_loader, num_classes, eps, device, normalize):
    attack = torchattacks.AutoAttack(model, eps=eps, n_classes=num_classes)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"AutoAttack (ε={eps})", device)

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

    accuracy = []
    
    acc, prec, rec, f1 = accuracy_clean(model, testLoader, device)
    accuracy.append(acc)
    all_eps = [0.01, 8/255, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
    #all_eps = [0.13, 0.15, 0.175, 0.2]

    for eps in all_eps:
        accuracy_FGSM(model, testLoader, eps, device, config.normalize)
    
    for eps in all_eps:
        accuracy_PGD(model, testLoader, eps, device, config.normalize)
    
    for eps in all_eps:
        accuracy_MIM(model, testLoader, eps, device, config.normalize)
    
    for eps in all_eps:
        accuracy_AutoAttack(model, testLoader, 4, eps, device, config.normalize)
    
    return True
