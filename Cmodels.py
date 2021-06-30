import torch
from torch import nn
from Cmodules import CLinear, CReLU, CModel, CCrossEntropyLoss


class CMLP3(CModel):
    def __init__(self):
        super().__init__()
        self.fc1 = CLinear(28*28,32)
        self.relu1 = CReLU()
        self.fc2 = CLinear(32,32)
        self.relu2 = CReLU()
        self.fc3 = CLinear(32,10)
    
    def Cbackward(self, output):
        with torch.no_grad():
            S = self.C_cross_entropy_back(output)
            S = self.fc3.Cbackward(S)
            S = self.relu2.Cbackward(S)
            S = self.fc2.Cbackward(S)
            S = self.relu1.Cbackward(S)
            S = self.fc1.Cbackward(S)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# class SMLP4(SModel):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = SLinear(28*28,32)
#         self.fc2 = SLinear(32,32)
#         self.fc3 = SLinear(32,32)
#         self.fc4 = SLinear(32,10)
#         self.relu = SReLU()
    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         xS = torch.zeros_like(x)
#         x, xS = self.fc1(x, xS)
#         x, xS = self.relu(x, xS)
#         x, xS = self.fc2(x, xS)
#         x, xS = self.relu(x, xS)
#         x, xS = self.fc3(x, xS)
#         x, xS = self.relu(x, xS)
#         x, xS = self.fc4(x, xS)
#         return x, xS


# class SLeNet(SModel):

#     def __init__(self):
#         super().__init__()
#         # 1 input image channel, 6 output channels, 3x3 square convolution
#         # kernel
#         self.conv1 = SConv2d(1, 6, 3, padding=1)
#         self.conv2 = SConv2d(6, 16, 3, padding=1)
#         # an affine operation: y = Wx + b
#         self.fc1 = SLinear(16 * 7 * 7, 120)  # 6*6 from image dimension
#         self.fc2 = SLinear(120, 84)
#         self.fc3 = SLinear(84, 10)
#         self.pool = SMaxpool2D(2)
#         self.relu = SReLU()

#     def forward(self, x):
#         xS = torch.zeros_like(x)
#         x, xS = self.conv1(x, xS)
#         x, xS = self.relu(x, xS)
#         x, xS = self.pool(x, xS)
        
#         x, xS = self.conv2(x, xS)
#         x, xS = self.relu(x, xS)
#         x, xS = self.pool(x, xS)
        
#         x = x.view(-1, self.num_flat_features(x))
#         if xS is not None:
#             xS = xS.view(-1, self.num_flat_features(xS))
        
#         x, xS = self.fc1(x, xS)
#         x, xS = self.relu(x, xS)
        
#         x, xS = self.fc2(x, xS)
#         x, xS = self.relu(x, xS)
        
#         x, xS = self.fc3(x, xS)
#         return x, xS

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
