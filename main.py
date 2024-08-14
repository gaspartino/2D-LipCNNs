import os
import warnings
from argparse import ArgumentParser
from train import *
from attack import *
warnings.filterwarnings("ignore")

def main(args):

    config = args 

    config.lip_batch_size = 64
    config.print_freq = 10
    config.save_freq = 5
    config.dataset == 'mnist'
    config.in_channels = 1
    config.img_size = 32
    config.num_classes = 10
    config.loss = 'xent'
    config.offset = 1.5

    if config.model == 'Lip2C2F':
        config.layer = 'LipC2F'
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
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    #parser.add_argument('--num_workers', type=int, default=4)
    #parser.add_argument('--LLN', action='store_true')
    #parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--cert_acc', action='store_true', default=True)
    
    args = parser.parse_args()

    #seeds = [1, 123, 296] #[39, 927, 834, 568]
    seeds = [123,296,1]
    #models = ['LipLeNet5', 'VanillaLeNet5','AllLeNet5']
    #models = ['LipLeNet5']

    #models = ['All2C2F', 'Lip2C2F']
    #models = ['Lip2C2FPool'] ['Vanilla2C2FPool', 'Vanilla2C2F']
    models = ['All2C2F', 'Lip2C2F', 'Lip2C2FPool', 'AOL2C2F']
    layers = ['Orthogon', 'Sandwich']
    #gammas = [1.0, 2.0]
    #gammas = [10.0, 20.0, 50.0, 100.0]
    #gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    gammas = [2.0, 4.0]

    for seed in seeds:
        args.seed = seed
        for model in models:
            args.model = model
            print(f"Running with seed: {seed}, model: {model}")
       
            # Check if the current model allows gamma settings
            if model in ['All2C2F', 'Lip2C2F','AOL2C2F','Lip2C2FPool']:
                for gamma in gammas:
                    args.gamma =gamma
                    print(f"Running with gamma: {gamma}")
                    if model == 'All2C2F':
                        for layer in layers:
                            args.layer = layer
                            print(f"Running with layer: {layer}")
                            main(args)
                    else:
                        #args.layer = 'Lip2C2F'
                        main(args)
                    # Call your function or perform actions here with seed, model, and gamma
            else:
                # Call your function or perform actions here with seed and model (no gamma)
                #args.layer = 'Vanilla2C2F'
                main(args)

    # for seed in seeds:
    #     args.seed = seed
    #     for model in models:
    #         args.model = model
    #         print(f"Running with seed: {seed}, model: {model}")
        
    #         # Check if the current model allows gamma settings
    #         if model in ['VanillaLeNet5']:
    #             args.layer = 'VanillaLeNet5'
    #             main(args)
    #         else:
    #             for gamma in gammas:
    #                 args.gamma =gamma
    #                 print(f"Running with gamma: {gamma}")
    #                 if model == 'AllLeNet5':
    #                     args.layer = 'Aol'
    #                     main(args)
    #                 else:
    #                     args.layer = 'LipLeNet5'
    #                     main(args)

    #for seed in seeds:
    #    args.seed = seed
    #    for gamma in gammas:
    #        args.gamma =gamma
    #        main(args)