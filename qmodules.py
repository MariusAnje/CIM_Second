import torch
from modules import SModule, NModule
from Functions import QuantFunction
from torch import nn
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction, SBatchNorm2dFunction

quant = QuantFunction.apply
_moving_momentum = 0.9

class QSLinear(SModule):
    def __init__(self, N, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply
        self.N = N
        if self.op.bias is not None:
            nn.init.zeros_(self.op.bias)
        nn.init.kaiming_normal_(self.op.weight)
        self.register_buffer('input_range', torch.zeros(1))
    
    def copy_N(self):
        new = QNLinear(self.N, self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.input_range = self.input_range
        new.original_w = self.original_w
        new.original_b = self.original_b
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, quant(self.N, self.op.weight) + self.noise, self.weightS)
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias)
        if self.op.bias is not None:
            xS += self.op.bias
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range), xS

class QSConv2d(SModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply
        self.N = N
        if self.op.bias is not None:
            nn.init.zeros_(self.op.bias)
        nn.init.kaiming_normal_(self.op.weight)
        self.register_buffer('input_range', torch.zeros(1))

    def copy_N(self):
        new = QNConv2d(self.N, self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.input_range = self.input_range
        new.original_w = self.original_w
        new.original_b = self.original_b
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, quant(self.N, self.op.weight) + self.noise, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias).reshape(1,-1,1,1).expand_as(x)
        if self.op.bias is not None:
            xS += self.op.bias.reshape(1,-1,1,1).expand_as(xS)
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range), xS

class QNLinear(NModule):
    def __init__(self, N, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.linear
        self.N = N
        self.scale = 1.0
        self.register_buffer('input_range', torch.zeros(1))

    def copy_S(self):
        new = QSLinear(self.N, self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.input_range = self.input_range
        return new

    def forward(self, x):
        x = self.function(x, quant(self.N,self.op.weight) + self.noise, None)
        x = x * self.scale
        if self.op.bias is not None:
            x += self.op.bias
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range)

class QNConv2d(NModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.conv2d
        self.N = N
        self.scale = 1.0
        self.register_buffer('input_range', torch.zeros(1))
    
    def copy_S(self):
        new = QSConv2d(self.N, self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        new.scale = self.scale
        new.input_range = self.input_range 
        return new

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.zero_()

    def forward(self, x):
        x = self.function(x, quant(self.N, self.op.weight) + self.noise, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x = x * self.scale
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias).reshape(1,-1,1,1).expand_as(x)
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range)

