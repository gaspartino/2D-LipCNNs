import os
import warnings
from argparse import ArgumentParser
from train import *
from attack import *
warnings.filterwarnings("ignore")

def main(args):

    config = args 

    config.lip_batch_size = 512
    config.print_freq = 10
    config.save_freq = 5
    config.dataset == 'bstl'
    config.in_channels = 3
    config.img_size = 32
    config.num_classes = 4
    config.loss = 'xent'
    config.offset = 1.5

    if config.model in ['Lip2C2F', 'Lip2C1F', 'Lip3C1F']:
        config.layer = 'Lip2C2F'
    elif config.model == 'Lip2C2FPool':
        config.layer = 'LipC2FPool'
    elif config.model == 'Vanilla2C2F':
        config.layer = 'Vanilla2C2F'
        config.gamma = None
    elif config.model == 'Vanilla2C2FPool':
        config.layer = 'Vanilla2C2FPool'
        config.gamma = None
    elif config.model == 'AOL2C2F':
        config.layer = 'AOL2C2F'
    elif config.model == 'LipLeNet5':
        config.layer = 'LipLeNet5'
    elif config.model == 'LipLeNet5Max':
        config.layer = 'LipLeNet5Max'
    elif config.model == 'VanillaLeNet5':
        config.layer = 'VanillaLeNet5'
        config.gamma = None
    elif config.model == 'FCModel':
        config.layer = 'FCModel'
        config.gamma = None

    if config.gamma is None:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.model}-{config.layer}"
    else:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.model}-{config.layer}-gamma{config.gamma:.1f}"

    os.makedirs("./data", exist_ok=True)
    os.makedirs(config.train_dir, exist_ok=True)
    if config.mode == 'train':
        train(config)
    elif config.mode == 'attack':
        PGDL2_attack(config)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-m', '--model', type=str, default='Lip2C2F',
                        help="[Lip2C2F, All2C2F, Vanilla2C2F, LipLeNet5, LipLeNet5Max, AllLeNet5, Vanilla2C2F]")
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help="Network Lipschitz bound") # 1.0
    parser.add_argument('-s', '--seed', type=int, default=1) # 123
    parser.add_argument('-e','--epochs', type=int, default=20) # 100
    parser.add_argument('--layer', type=str, default='Aol')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--root_dir', type=str, default='./saved_models')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('-d', '--dataset', type=str, default='bstl')
    parser.add_argument('--cert_acc', action='store_true', default=True)
    parser.add_argument('--normalize', action='store_true', default=False)
    
    args = parser.parse_args()

    seeds = [1]

    models = ['Lip2C2F']
    layers = ['Lip2C2F']

    gammas = [1.0]

    for seed in seeds:
        args.seed = seed
        for model in models:
            args.model = model
            print(f"Running with seed: {seed}, model: {model}")
       
            if model in ['All2C2F', 'Lip2C2F','AOL2C2F','Lip2C2FPool']:
                for gamma in gammas:
                    args.gamma = gamma
                    print(f"Running with gamma: {gamma}")
                    if model == 'All2C2F':
                        for layer in layers:
                            args.layer = layer
                            print(f"Running with layer: {layer}")
                            main(args)
                    else:
                        main(args)
            else:
                main(args)
