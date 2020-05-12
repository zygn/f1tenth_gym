from datasets import SteerDataset_Class
from models import NVIDIA_ConvNet
import os, pickle, random, time
import matplotlib.pyplot as plt
import pdb

#torch imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

#Logging
from tensorboardX import SummaryWriter
train_writer = SummaryWriter(logdir="../logs")

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"
FOLDERPATH = "../data/sim_train"
KNN_MODELPATH = "knn_model"

def seed_env():
    seed = 6582 
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 


def get_dataloader(dataset, bs):
    vsplit = 0.0 #Ideally have this as an argument

    if vsplit == 0.0:
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return train_dataloader, None

    dset_size = len(dataset)
    idxs = list(range(dset_size))
    split = int(np.floor(vsplit * dset_size))
    np.random.shuffle(idxs)
    train_idxs, val_idxs = idxs[split:], idxs[:split]

    #Using SubsetRandomSampler but should ideally sample equally from each steer angle to avoid distributional bias
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    train_dataloader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset, batch_size=bs, sampler=val_sampler)
    return train_dataloader, valid_dataloader

def pairwise_loss(features_1, features_2, same_class):
  features_diff = features_1 - features_2
  diffs_squared_sum = torch.sum(torch.mul(features_diff, features_diff), dim=-1)
  m = 2.0
  # Invariant is that same_class is 1 when they are the same class, 0 otherwise
  losses = torch.where((same_class[:, 0] == 1.0), diffs_squared_sum, torch.max(torch.tensor([0.], device=device), m - diffs_squared_sum))
  return torch.sum(losses)

def loss_pass(net, loss_func, loader1, loader2, epoch, optim, op='train'):
    print("{op} epoch: {epoch}".format(op=op, epoch=epoch)) 
    t0 = time.time()
    total_epoch_loss = 0
    for i, data in enumerate(zip(loader1, loader2)):
        #1) Extract data
        d1, d2 = data
        ts_imgbatch1, ts_imgbatch2 = d1.get("img").to(device), d2.get("img").to(device)
        ts_tgtbatch1, ts_tgtbatch2 = d1.get("target").to(device), d2.get("target").to(device)
        ts_same_class = (ts_tgtbatch1 == ts_tgtbatch2).float()


        #2) Subsample 2:1 ratio of negative to positive samples
        all_pos_idxs = (ts_same_class == 1.0).nonzero()
        all_neg_idxs = (ts_same_class == 0.0).nonzero()
        pos_count = all_pos_idxs.shape[0]
        neg_count = pos_count*2
        if pos_count == 0:
            continue

        ts_pos_img1 = ts_imgbatch1[all_pos_idxs[:, 0]]
        ts_neg_img1 = ts_imgbatch1[all_neg_idxs[:, 0]]
        ts_neg_img1 = ts_neg_img1[0:neg_count]

        ts_pos_img2 = ts_imgbatch1[all_pos_idxs[:, 0]]
        ts_neg_img2 = ts_imgbatch2[all_neg_idxs[:, 0]]
        ts_neg_img2 = ts_neg_img1[0:neg_count]

        ts_pos_class = ts_same_class[all_pos_idxs[:, 0]]
        ts_neg_class = ts_same_class[all_neg_idxs[:, 0]]
        ts_neg_class = ts_neg_class[0:neg_count]

        ts_imgs1 = torch.cat((ts_pos_img1, ts_neg_img1), dim=0)
        ts_imgs2 = torch.cat((ts_pos_img2, ts_neg_img2), dim=0)

        ts_classes = torch.cat((ts_pos_class, ts_neg_class), dim=0)

        #3) Classic Training Loop
        optim.zero_grad()
        ts_features1 = net(ts_imgs1)
        ts_features2 = net(ts_imgs2)

        ts_loss = pairwise_loss(ts_features1, ts_features2, ts_classes)
        if op=='train':
            ts_loss.backward()
            optim.step()
        print("loss:{}".format(ts_loss.item())) 
        total_epoch_loss += ts_loss.item()
        if i % 20 == 0:
            #do some interesting visualization of results here
            pass
    print("FINISHED {op} EPOCH{epoch}".format(op=op, epoch=epoch))
    print("----{now} seconds----".format(now=time.time()-t0, op=op, epoch=epoch))            
    return total_epoch_loss

seed_env()

# 1: Load Dataset, split into train & val
print(f"Loading Dataset from {FOLDERPATH} ...")
knn_model = pickle.load(open(KNN_MODELPATH, 'rb'))
dset = SteerDataset_Class(FOLDERPATH, knn_model)
train_loader1, _ = get_dataloader(dset, 32)
train_loader2, _ = get_dataloader(dset, 32)
d = dset[0]

# 2: Get Model, Optimizer, Loss Function & Num Epochs
net = NVIDIA_ConvNet(args_dict={"fc_shape":64*23*33}).to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
num_epochs = int(1e+4)

train_losses = []

# 3: TRAIN: Main Training Loop over epochs
print(f"STARTING FROM EPOCH: {0}")
best_train_loss = float('inf')
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    print(f"Starting Epoch:{epoch}")
    train_epoch_loss = loss_pass(net, loss_func, train_loader1, train_loader2, epoch, optim, op='train')
    print("----------------EPOCH{}STATS:".format(epoch))
    print("TRAIN LOSS:{}".format(train_epoch_loss))
    if best_train_loss > train_epoch_loss:
        best_train_loss = train_epoch_loss
        torch.save(net.state_dict(), "train_sim_net")
    train_writer.add_scalar("Loss", train_epoch_loss, epoch)
    train_losses.append(train_epoch_loss)
    if epoch % 4 == 0:
        pickle.dump(train_losses, open("train_losses.pkl", "wb"))