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
import torch
import numpy as np
from torch import nn
from Functions import SCrossEntropyLossFunction
from modules import SReLU, SModel, SMaxpool2D, SModule, NModule
from qmodules import QSLinear, QSConv2d
from qmodels import QSModel

class GQCIFAR(QSModel):
    def __init__(self, rollout, kL, cL, bL):
        super().__init__()

        kS = kL[rollout[0]]
        pD = kS // 2
        ch = cL[rollout[1]]
        N  = bL[rollout[2]]

        self.conv1 = QSConv2d(N, 3,     ch[0], kS, padding=pD)
        self.conv2 = QSConv2d(N, ch[0], ch[0], kS, padding=pD)
        self.pool1 = SMaxpool2D(2,2)

        self.conv3 = QSConv2d(N, ch[0], ch[1], kS, padding=pD)
        self.conv4 = QSConv2d(N, ch[1], ch[1], kS, padding=pD)
        self.pool2 = SMaxpool2D(2,2)

        self.conv5 = QSConv2d(N, ch[1], ch[2], kS, padding=pD)
        self.conv6 = QSConv2d(N, ch[2], ch[2], kS, padding=pD)
        self.pool3 = SMaxpool2D(2,2)
        
        self.fc1 = QSLinear(N, ch[2] * 4 * 4, 1024)
        self.fc2 = QSLinear(N, 1024, 1024)
        self.fc3 = QSLinear(N, 1024, 10)
        self.relu = SReLU()

    def forward(self, x):
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.unpack_flattern(x)
 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def CEval():
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEval(dev_var, write_var):
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var, write_var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NEachEval(dev_var, write_var):
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().numpy()

def NTrain(epochs, header, dev_var, write_var, verbose=False):
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = NEachEval(dev_var, write_var)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def RecoverBN(epoch):
    model.train()
    model.clear_noise()
    for _ in range(epoch):
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs, outputsS = model(images)

def GetSecond():
    model.eval()
    model.clear_noise()
    optimizer.zero_grad()
    # for images, labels in tqdm(secondloader):
    for images, labels in secondloader:
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1, 784)
        outputs, outputsS = model(images)
        loss = criteria(outputs, outputsS,labels)
        loss.backward()
        # sail_list = None
        # for m in model.modules():
        #     if isinstance(m, SModule):
        #         sail = m.mask_indicator("second", 0).view(-1)
        #         if sail_list is None:
        #             sail_list = sail
        #         else:
        #             sail_list = torch.cat([sail_list, sail])
        # import time
        # torch.save(sail_list, f"S_grad_{time.time()}.pt")
        # exit()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

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
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--each_only', action='store', type=str2bool, default=False,
            help='only use noise each, do not MC')        
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
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
    args = parser.parse_args()

    print(args)
    # ch1L = [ 6, 12, 24]
    # qb1L = [2, 3, 4, 5]
    # qb2L = [2, 3, 4, 5]
    # ch2L = [16, 32, 64]
    # kN1L = [1, 3, 5, 7]
    # kN2L = [1, 3, 5, 7]

    kL = [1, 3, 5]
    cL = [(8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)]
    bL = [2, 4, 6, 8]

    space = [   kL,
                cL,
                bL]

    BS = 128
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        normalize])
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
            ])
    trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
    secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)

    header = time.time()
    header_timer = header
    total = 1
    for i in range(len(space)):
        total *= len(space[i])
    for i in range(total):
        k = i
        rolloutI = []
        for j in range(len(space)):
            rolloutI.append(k % len(space[j]))
            k = k // len(space[j])

        print("rollout: ", rolloutI)
        
        model = GQCIFAR(rolloutI, space[0], space[1], space[2])
        parent_path = "./"
        
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.push_S_device()
        model.clear_noise()
        model.clear_mask()
        criteria = SCrossEntropyLoss()
        criteriaF = torch.nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])

        model.to_first_only()
        NTrain(args.train_epoch, header, args.train_var, 0.0, args.verbose)
        if args.train_var > 0:
            state_dict = torch.load(f"tmp_best_{header}.pt")
            model.load_state_dict(state_dict)
        model.from_first_back_second()
        torch.save(model.state_dict(), f"saved_B_{header}.pt")
        state_dict = torch.load(f"saved_B_{header}.pt")
        print(f"No mask no noise: {CEval():.4f}")
        test_acc = NEachEval(args.dev_var, args.write_var)
        print(f"No mask noise each: {test_acc:.4f}")
        model.load_state_dict(state_dict)
        model.clear_mask()
        model.back_real(device)
        model.push_S_device()

        GetSecond()
        print(f"S grad before masking: {model.fetch_S_grad().item():E}")

        if not args.each_only:
            no_mask_acc_list = []
            loader = range(args.noise_epoch)
            for _ in loader:
                acc = NEval(args.dev_var, 0.0)
                no_mask_acc_list.append(acc)
            print(f"No mask noise average acc: {np.mean(no_mask_acc_list):.4f}, std: {np.std(no_mask_acc_list):.4f}")
            torch.save(no_mask_acc_list, f"no_mask_list_{header}_{args.dev_var}.pt")
    
