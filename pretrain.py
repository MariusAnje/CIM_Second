import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import numpy as np
from models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss
import modules
from qmodels import QSLeNet, QCIFAR
import resnet
import qresnet
import qvgg
import qdensnet
from modules import SModule
from tqdm import tqdm
import time
import argparse
import os
from utils import CEval, NEachEval, NEval, CTrain, RecoverBN, GetSecond, str2bool
from utils import get_dataset, get_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG"],
            help='model to use')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--div', action='store', type=int, default=1,
            help='division points for second')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='if to use tqdm')
    args = parser.parse_args()

    print(args)
    header = time.time()
    header_timer = header
    parent_path = "./"
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainloader, testloader_file, secondloader = get_dataset(args)
    testloader = []
    for i in testloader_file:
        testloader.append(i)
    model, optimizer, scheduler, criteria, criteriaF = get_model(args, device)

    model_group = model, scheduler, criteriaF, optimizer, trainloader, testloader, device
    model.to_first_only()
    CTrain(model_group, args.train_epoch, header, args.train_var, 0.0, args.verbose)
    if args.train_var > 0:
        state_dict = torch.load(f"tmp_best_{header}.pt")
        model.load_state_dict(state_dict)
    model.from_first_back_second()
    torch.save(model.state_dict(), f"saved_B_{header}.pt")
    state_dict = torch.load(f"saved_B_{header}.pt")
    print(f"No mask no noise: {CEval():.4f}")
    model.load_state_dict(state_dict)
    model.clear_mask()
