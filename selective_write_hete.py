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
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.0,
            help='device variation [std] when training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--write_var', action='store', type=float, default=0.03,
            help='device variation [std] after write and verify')
    parser.add_argument('--mask_p', action='store', type=float, default=0.01,
            help='portion of the mask')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG"],
            help='model to use')
    parser.add_argument('--method', action='store', default="SM", choices=["second", "magnitude", "saliency", "random", "SM", "HSM"],
            help='method used to calculate saliency')
    parser.add_argument('--alpha', action='store', type=float, default=1e6,
            help='weight used in saliency - substract')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
    parser.add_argument('--use_mask', action='store',type=str2bool, default=True,
            help='if to do the masking experiment')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    parser.add_argument('--calc_S', action='store',type=str2bool, default=True,
            help='if calculated S grad if not necessary')
    parser.add_argument('--div', action='store', type=int, default=1,
            help='division points for second')
    parser.add_argument('--layerwise', action='store',type=str2bool, default=False,
            help='if do it layer by layer')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='if to use tqdm')
    parser.add_argument('--s_rate', action='store',type=float, default=1.,
            help='rate of device var')
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
    if not args.pretrained:
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

        no_mask_acc_list = []
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(model_group, args.dev_var, 0.0, args.s_rate)
            no_mask_acc_list.append(acc)
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.dev_var}.pt")

        no_mask_acc_list = []
        loader = range(args.noise_epoch)
        for _ in loader:
            acc = NEval(model_group, args.write_var, 0.0, args.s_rate)
            no_mask_acc_list.append(acc)
        print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.write_var}.pt")

        exit()
    else:
        parent_path = args.model_path
        header = args.header
        try:
            no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.dev_var}.pt"))
            print(f"[{args.dev_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        except:
            print(f"[{args.dev_var}] Not Found")
        try:
            no_mask_acc_list = torch.load(os.path.join(parent_path, f"no_mask_list_{header}_{args.write_var}.pt"))
            print(f"[{args.write_var}] No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
        except:
            print(f"[{args.write_var}] Not Found")
        model.back_real(device)
        model.push_S_device()

    
    state_dict = torch.load(os.path.join(parent_path, f"saved_B_{header}.pt"), map_location=device)
    # model.to_first_only()
    model.load_state_dict(state_dict)
    # model.from_first_back_second()
    model.back_real(device)
    model.push_S_device()
    criteria = SCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20])
    model.clear_noise()

    model.normalize()
    model_group = model, scheduler, criteriaF, optimizer, trainloader, testloader, device
    print(f"No mask no noise: {CEval(model_group):.4f}")
    GetSecond(model_group, secondloader, criteria, args)
    print(f"S grad before masking: {model.fetch_S_grad().item():E}")
    # if "Res18" in args.model or "TIN" in args.model:
    #     model.fine_S_grad()
    model.fine_S_grad()
    
    if args.use_mask:
        model.clear_mask()
        mask_acc_list = []
        th = model.calc_sail_th(args.mask_p, args.method, args.alpha)
        model.set_mask_sail(th, "th", args.method, args.alpha)
        print(th)

        total, RM_new = model.get_mask_info()
        print(f"Weights removed: {RM_new/total:f}")
        model.de_normalize()
        print(f"S grad after  masking: {model.fetch_S_grad().item():E}")
        fine_mask_acc_list = []
        print(f"Finetune no noise: {CEval(model_group):.4f}")
        loader = range(args.noise_epoch)
        for _ in tqdm(loader):
            acc = NEval(model_group, args.dev_var, args.write_var, args.s_rate)
            fine_mask_acc_list.append(acc)
        print(f"Finetune noise average acc: {np.mean(fine_mask_acc_list):.4f}, std: {np.std(fine_mask_acc_list):.4f}")
