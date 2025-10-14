import torch 
import numpy as np
import time
from model import getModel
from dataset import getDataLoader
from utils import *
import csv
from attack import *
from compute_lip_2 import *

def train(config):
    
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getModel(config).to(device)
    criterion = getLoss(config)

    txtlog = TxtLogger(config)
    txtlog(vars(config))

    txtlog(f"Set global seed to {config.seed:d}")
    
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        txtlog(f"name: {config.model}-{config.layer}, num_params: {1e-6*nparams:.1f}M")
    else:
        txtlog(f"name: {config.model}-{config.layer}, num_params: {1e-3*nparams:.1f}K")
    
    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)
    PrintFreq = config.print_freq
    gamma = config.gamma

    opt = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]

    tloss_step, tacc_step, lr_step = [], [], []
    ttime_epoch,vtime_epoch,tloss_epoch,vloss_epoch,tacc_epoch, vacc_epoch, vacc_epoch = [],[],[],[],[],[],[]

    ind = 0
    if gamma == None:
        ind = 1

    for epoch in range(Epochs):
        ## train_step
        start = time.time()
        n, Loss, Acc = 0, 0.0, 0.0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            x, y = batch[0], batch[1]
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            opt.param_groups[0].update(lr=lr)

            yh = model(x)

            J = criterion(yh, y)

            opt.zero_grad()
            J.backward()
            opt.step()

            loss = J.item()
            n += y.size(0)
            Loss += loss * y.size(0)
            acc = (yh.max(1)[1] == y).sum().item()
            acc = 100*acc / y.size(0)
            Acc += acc * y.size(0)
            
            tloss_step.append(loss)
            tacc_step.append(acc)
            lr_step.append(lr)

            if (batch_idx+1) % PrintFreq == 0:
                print(f"Epoch: {epoch+1:3d} | {batch_idx+1:3d}/{steps_per_epoch}, acc: {Acc/n:.1f}, loss: {Loss/n:.2f}, lr: {100*lr:.3f}", end='\r', flush=True)
        
        train_time = time.time()-start 
        train_loss = Loss/n 
        train_acc = Acc/n

        ## dummy call to flush the new model parameter in the last batch
        model(torch.rand((1,x.shape[1], x.shape[2], x.shape[3])).to(x.device)) 

        n, Loss, Acc = 0, 0.0, 0.0
        Acc36, Acc72, Acc108 = 0.0, 0.0, 0.0
        Acc1, Acc2, Acc3, Acc1_58 = 0.0, 0.0, 0.0, 0.0
        model.eval()
#        if LLN:
#            #last_weight = model.model[-1].weight
#            last_weight = model.LipCNNFc2.weight
#            normalized_weight = torch.nn.functional.normalize(last_weight, p=2, dim=1)

        if ind == 1:
            gamma = lipschitz_upper_bound(model)

        start = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                Loss += criterion(yh, y).item() * y.size(0)
                n += y.size(0)
                correct = yh.max(1)[1] == y
                acc = correct.sum().item()
                Acc += acc

                if config.cert_acc and epoch == Epochs-1:
                    margins, indices = torch.sort(yh, 1)
                    cert36 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 36/255.0
                    cert72 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 72/255.0
                    cert108= (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma *108/255.0
                    cert1 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 1.0
                    cert2 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 2.0
                    cert3 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 3.0
                    cert1_58 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 1.58

                    Acc1 += torch.sum(correct & cert1).item()
                    Acc2 += torch.sum(correct & cert2).item()
                    Acc3 += torch.sum(correct & cert3).item()
                    Acc1_58 += torch.sum(correct & cert1_58).item()



        test_time = time.time()-start 
        test_loss = Loss/n
        test_acc = 100*Acc/n

        ttime_epoch.append(train_time)
        vtime_epoch.append(test_time)
        tloss_epoch.append(train_loss)
        vloss_epoch.append(test_loss)
        tacc_epoch.append(train_acc)
        vacc_epoch.append(test_acc)


        txtlog(f"Epoch: {epoch+1:3d} | time: {train_time:.1f}/{test_time:.1f}, loss: {train_loss:.2f}/{test_loss:.2f}, acc: {train_acc:.1f}/{test_acc:.1f}, 100lr: {100*lr:.3f}")

        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")

    Acc36 = 100.0*Acc36/n 
    Acc72 = 100.0*Acc72/n
    Acc108= 100.0*Acc108/n
    Acc1 = 100.0*Acc1/n 
    Acc2 = 100.0*Acc2/n
    Acc3= 100.0*Acc3/n
    Acc1_58= 100.0*Acc1_58/n
    print(f"Epsilon: 36 \tAccuracy: {Acc36:.4f}")
    print(f"Epsilon: 72 \tAccuracy: {Acc72:.4f}")
    print(f"Epsilon: 108 \tAccuracy: {Acc108:.4f}")

    # after training
    np.savetxt(f'{config.train_dir}/tloss_step.csv',np.array(tloss_step))
    np.savetxt(f'{config.train_dir}/tacc_step.csv',np.array(tacc_step))
    np.savetxt(f'{config.train_dir}/lr_step.csv',np.array(lr_step))
    np.savetxt(f'{config.train_dir}/ttime_epoch.csv',np.array(ttime_epoch))
    np.savetxt(f'{config.train_dir}/vtime_epoch.csv',np.array(vtime_epoch))
    np.savetxt(f'{config.train_dir}/tloss_epoch.csv',np.array(tloss_epoch))
    np.savetxt(f'{config.train_dir}/vloss_epoch.csv',np.array(vloss_epoch))
    np.savetxt(f'{config.train_dir}/tacc_epoch.csv',np.array(tacc_epoch))
    np.savetxt(f'{config.train_dir}/vacc_epoch.csv',np.array(vacc_epoch))

    xshape = (config.lip_batch_size, config.in_channels, config.img_size, config.img_size)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape))
    gam = empirical_lipschitz(model, x)
    if gamma is None:
        txtlog(f"Lipschitz: {gam:.2f}/--")
    else:
        txtlog(f"Lipschitz capcity: {gam:.4f}/{gamma:.2f}, {100*gam/gamma:.2f}")

    adv_accuracies = PGDL2_attack(config)

    filename = "saved_models.csv"
    data = [config.model, config.layer, config.seed, config.epochs, gamma, gam, test_acc, adv_accuracies[0].item(), adv_accuracies[1].item(), adv_accuracies[2].item(), adv_accuracies[3].item(), adv_accuracies[4].item(), Acc36, Acc72, Acc108]

    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
    
        # Write the new data line
        writer.writerow(data)

def train_toy(config):
    
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config)
    criterion = getLoss(config)

    txtlog = TxtLogger(config)
    # wanlog = WandbLogger(config)

    txtlog(f"Set global seed to {config.seed:d}")
    
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")
    
    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)
    gamma = config.gamma

    opt = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]

    for epoch in range(Epochs):
        ## train_step
        n, Loss = 0, 0.0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            x, y = batch[0], batch[1]
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            opt.param_groups[0].update(lr=lr)

            yh = model(x)
            J = criterion(yh, y)

            opt.zero_grad()
            J.backward()
            opt.step()

            loss = J.item()
            n += y.size(0)
            Loss += loss * y.size(0)
        
        train_loss = Loss/n 

        ## dummy call to flush the new model parameter in the last batch
        model(torch.rand((1,x.shape[1])).to(x.device)) 

        n, Loss = 0, 0.0, 
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0], batch[1]
                yh = model(x)
                Loss += criterion(yh, y).item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n

        txtlog(f"Epoch: {epoch+1:3d} | loss: {train_loss:.2f}/{test_loss:.2f}, 100lr: {100*lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")
