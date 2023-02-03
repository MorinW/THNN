import json
import os
import time
import copy
import torch
import torch.optim as optim
import utils.hypergraph_utils as hgut
from datasets import load_feature_construct_H
from models import THNN_ab, HyperGraph
from tqdm import tqdm
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='my description')


data_dir_modelnet = './data/features/ModelNet40_mvcnn_gvcnn.mat'
data_dir_modelnut2012 = './data/features/NTU2012_mvcnn_gvcnn.mat'
parser.add_argument('--dataset', default='NUT2012')
parser.add_argument('--m', default=True)
parser.add_argument('--g', default=True)
parser.add_argument('--ms', default=True)
parser.add_argument('--gs', default=True)
parser.add_argument('--probH', default=False)
parser.add_argument('--rank', default=32)
parser.add_argument('--knei', default=4)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--weight_decay', default=0.00005)
parser.add_argument('--n_hid', default=128)
parser.add_argument('--print_freq', default=1)
parser.add_argument('--max_epoch', default=500)
parser.add_argument('--drop_out', default=0)
parser.add_argument('--file_name', default='./log/log.txt')
parser.add_argument('--mode', default=0)
args = parser.parse_args()
print(type(args.gs), bool(int(args.gs)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


on_dataset = args.dataset
use_mvcnn_feature = bool(int(args.m))
use_gvcnn_feature = bool(int(args.g))
use_mvcnn_feature_for_structure = bool(int(args.ms))
use_gvcnn_feature_for_structure = bool(int(args.gs))
is_probH = bool(int(args.probH))
rank = int(args.rank)
K_neigs = int(args.knei)
n_hid = int(args.n_hid)
max_epoch = int(args.max_epoch)
print_freq = int(args.print_freq)
lr = float(args.lr)
weight_decay = float(args.weight_decay)
drop_out = float(args.drop_out)
file_name = args.file_name
mode = int(args.mode)

# print(mode, file_name)

# initialize data
data_dir = data_dir_modelnut2012 if on_dataset == 'NUT2012' \
    else data_dir_modelnet
fts, lbls, idx_train, idx_test, H = \
    load_feature_construct_H(data_dir,
                             m_prob=1.0,
                             K_neigs=K_neigs,
                             is_probH=is_probH,
                             use_mvcnn_feature=use_mvcnn_feature,
                             use_gvcnn_feature=use_gvcnn_feature,
                             use_mvcnn_feature_for_structure=use_mvcnn_feature_for_structure,
                             use_gvcnn_feature_for_structure=use_gvcnn_feature_for_structure)

n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)
H = torch.bernoulli(torch.tensor(H)).numpy()
HyperGraph_mine = HyperGraph(H)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(HyperGraph_mine, fts)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            with open(file_name, 'a+') as f:
                f.write(f'Best val Acc: {best_acc:4f}')
                f.write('-' * 20)
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    with open(file_name, 'a+') as f:
        f.write(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        f.write(f'Best val Acc: {best_acc:4f}')
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _main():
    print(f"Classification on {on_dataset} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {use_mvcnn_feature}")
    print(f"use GVCNN feature: {use_gvcnn_feature}")
    print(f"use MVCNN feature for structure: {use_mvcnn_feature_for_structure}")
    print(f"use GVCNN feature for structure: {use_gvcnn_feature_for_structure}")
    print('Configuration -> Start')
    print(f"probH: {is_probH}")
    print(f"rank: {rank}")
    print(f"K_neigs: {K_neigs}")
    print(f"n_hid: {n_hid}")
    print(f"max_epoch: {max_epoch}")
    print(f"print_freq: {print_freq}")  
    print(f"lr: {lr}")
    print(f"weight_decay: {weight_decay}")
    print(f"drop_out: {drop_out}")  
    print(f"mode: {mode}") 
    print('Configuration -> End')
    
    with open(file_name, 'a+') as f:
        f.write('\n\nSettings:\n')
        f.write(f"Classification on {on_dataset} dataset!!! class number: {n_class}")
        f.write(f"use MVCNN feature: {use_mvcnn_feature}")
        f.write(f"use GVCNN feature: {use_gvcnn_feature}")
        f.write(f"use MVCNN feature for structure: {use_mvcnn_feature_for_structure}")
        f.write(f"use GVCNN feature for structure: {use_gvcnn_feature_for_structure}")
        f.write('Configuration -> Start')
        f.write(f"probH: {is_probH}")
        f.write(f"rank: {rank}")
        f.write(f"K_neigs: {K_neigs}")
        f.write(f"n_hid: {n_hid}")
        f.write(f"max_epoch: {max_epoch}")  
        f.write(f"lr: {lr}")
        f.write(f"weight_decay: {weight_decay}")
        f.write(f"mode: {mode}") 
        f.write(f"drop_out: {drop_out}")  
        f.write('Configuration -> End') 
        f.write('Training Procedure:\n')

    model_ft = THNN_ab(
        featdim = fts.shape[1],
        hiddendim = n_hid,
        outputdim = n_hid,
        rank = rank,
        n_class = n_class,
        dropout=drop_out,
        mode = mode)
    
    model_ft = model_ft.to(device)
    
    optimizer = optim.Adam(model_ft.parameters(), lr, weight_decay = weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0) 
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer, schedular, max_epoch, print_freq = print_freq)


if __name__ == '__main__':
    _main()
