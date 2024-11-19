import torch
import torch.linalg as la
import numpy as np
import torchattacks
from torchvision.transforms import Normalize
from autoattack import AutoAttack
from model import getModel
from dataset import getDataLoader
from utils import *
import matplotlib.pyplot as plt


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

def PGDL2_attack(config):
    #seed_everything(config.seed)
    #model = getModel(config)
    #_, testLoader = getDataLoader(config)
    #model_state = torch.load(f"{config.train_dir}/model.ckpt")
    #model.load_state_dict(model_state, strict = False)

    #xshape = (1, config.in_channels, config.img_size, config.img_size)
    #x = (torch.rand(xshape) + 0.3*torch.randn(xshape)) #.cuda()
    #model(x) # allocate memory for Q param

    #model.eval()

    seed_everything(config.seed)
    model = getModel(config) #.cuda() 
    _, testLoader = getDataLoader(config)
    txtlog = TxtLogger(config)
    xshape = (1, config.in_channels, config.img_size, config.img_size)
    x = torch.rand(xshape) #+ 0.3*torch.randn(xshape)) #.cuda()
    model(x) # allocate memory for Q param
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    model(x) # update Q param

    num_classes = len(set(testLoader.dataset.targets.numpy()))  # Acessando as classes no dataset MNIST
    print(f'Número de classes: {num_classes}')

    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #epsilons = [0.0, 1.0, 2.0, 3.0, 1.58]
    accuracies = []

    for epsilon in epsilons:
        pgdl2 = torchattacks.PGDL2(model, eps = epsilon)
        bim = torchattacks.BIM(model, eps=epsilon, alpha=2/255, steps=10)
        fgsm = torchattacks.FGSM(model, eps=epsilon)
        pgd = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10, random_start=True)

        correct = 0
        total = 0

        for images, labels in testLoader:
            #if epsilon == 0:
            #    adv_images = images
            #else:
            adv_images = pgdl2(images, labels)
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()
            total += labels.size(0)

        accuracy = correct / total

        accuracies.append(accuracy)
        print(f"PGDL2 | Epsilon: {epsilon}\tAccuracy: {accuracy:.4f}")
        
        correct = 0
        total = 0

        for images, labels in testLoader:
            #if epsilon == 0:
            #    adv_images = images
            #else:
            adv_images = pgd(images, labels)
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()
            total += labels.size(0)

        accuracy = correct / total

        accuracies.append(accuracy)
        print(f"PGD | Epsilon: {epsilon}\tAccuracy: {accuracy:.4f}")

        correct = 0
        total = 0

        for images, labels in testLoader:
            #if epsilon == 0:
            #    adv_images = images
            #else:
            adv_images = fgsm(images, labels)
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()
            total += labels.size(0)

        accuracy = correct / total

        accuracies.append(accuracy)
        print(f"FGSM | Epsilon: {epsilon}\tAccuracy: {accuracy:.4f}")

        correct = 0
        total = 0

        for images, labels in testLoader:
            #if epsilon == 0:
            #    adv_images = images
            #else:
            adv_images = bim(images, labels)
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()
            total += labels.size(0)

        accuracy = correct / total

        accuracies.append(accuracy)
        print(f"BIM | Epsilon: {epsilon}\tAccuracy: {accuracy:.4f}")

        print("")
    #plot_accuracies(epsilons, accuracies)
    return accuracies
