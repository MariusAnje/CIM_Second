import torch
from torch import nn
import torch.nn.functional as F
# from torch._C import device
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction    
from TFunctions import TLinearFunction, TConv2dFunction, TMSEFunction, TCrossEntropyLossFunction

class NMMModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightH = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)

    def set_noise(self, var):
        self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightH.grad.abs().view(-1).quantile(1-portion)
            self.mask = (self.weightH.grad.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.weightH.grad.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)
    
    def push_H_device(self):
        self.weightH = self.weightH.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)

    def clear_H_grad(self):
        with torch.no_grad():
            if self.weightH.grad is not None:
                self.weightH.grad.data *= 0
    
    def fetch_H_grad(self):
        return (self.weightH.grad.abs() * self.mask).sum()
    
    def fetch_H_grad_list(self):
        return (self.weightH.grad.data * self.mask)

class OModule(NMMModule):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)

    def from_others(self, source):
        pass

class SModule(NMMModule):
    def __init__(self):
        super().__init__()
    
    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightH.grad.data + 1e-10)

class TModule(NMMModule):
    def __init__(self):
        super().__init__()

    def do_third(self, alpha):
        assert(alpha <= 1 and alpha >= 0)
        self.op.weight.grad.data = (1 - alpha) * self.op.weight.grad.data + alpha * self.weightT.grad.data

class OLinear(OModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = F.linear

    def forward(self, x, xS):
        x = self.function(x, (self.op.weight + self.noise) * self.mask, self.op.bias)
        return x, None

class SLinear(SModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply

    def forward(self, x, xS):
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightH, self.op.bias)
        return x, xS

class TLinear(TModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = TLinearFunction.apply

    def forward(self, x, xT):
        x, xT = self.function(x, xT, (self.op.weight + self.noise) * self.mask, self.weightT, self.op.bias)
        return x, xT

class OConv2d(OModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = F.conv2d

    def forward(self, x, xS):
        x = self.function(x, (self.op.weight + self.noise) * self.mask, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x, None

class SConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def forward(self, x, xS):
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightH, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x, xS

class TConv2d(TModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = TConv2dFunction.apply

    def forward(self, x, xT):
        x, xT = self.function(x, xT, (self.op.weight + self.noise) * self.mask, self.weightT, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x, xT

class OReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU()
    
    def forward(self, x, xS):
        return self.op(x), None

class SReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU()
    
    def forward(self, x, xS):
        return self.op(x), self.op(xS)

class TReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU()
    
    def forward(self, x, xT):
        return self.op(x), self.op(xT)

class OMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def forward(self, x, xS):
        x, indices = self.op(x)
        return x, None

class SMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, x, xS):
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xS = xS.view(shape)[indices].view(x.shape)
        return x, xS

class TMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, x, xT):
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xT = xT.view(shape)[indices].view(x.shape)
        return x, xT

class NMMModule(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
    
    def push_H_device(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.push_H_device()

    def clear_H_grad(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_H_grad()

    def fetch_H_grad(self):
        H_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                H_grad_sum += m.fetch_S_grad()
        return H_grad_sum
    
    def calc_H_grad_th(self, quantile):
        H_grad_list = None
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                if H_grad_list is None:
                    H_grad_list = m.fetch_H_grad_list().view(-1)
                else:
                    H_grad_list = torch.cat([H_grad_list, m.fetch_H_grad_list().view(-1)])
        th = torch.quantile(H_grad_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, var):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.set_noise(var)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_noise()
    
    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.set_mask(th, mode)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d) or isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_mask()
    
    def do_third(self, alpha):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.do_third(alpha)
    
    def do_second(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.do_second()

    def to_first(self):
        for name, m in self.named_modules():
            if isinstance(m, SLinear) or isinstance(m, TLinear):
                if m.op.bias is not None:
                        bias = True
                new_layer = OLinear(m.op.in_features, m.op.out_features, bias)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, SConv2d) or isinstance(m, TConv2d):
                if m.op.bias is not None:
                        bias = True
                new_layer = OConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, SMaxpool2D) or isinstance(m, TMaxpool2D):
                    new_layer = OMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new_layer.op = m.op
                    new_layer.op.return_indices = True
                    self._modules[name] = new_layer

            elif isinstance(m, SReLU) or isinstance(m, TReLU):
                new_layer = OReLU()
                new_layer.op = m.op
                self._modules[name] = new_layer

            else:
                raise NotImplementedError
    
    def to_second(self):
        for name, m in self.named_modules():
            if isinstance(m, OLinear) or isinstance(m, TLinear):
                if m.op.bias is not None:
                        bias = True
                new_layer = SLinear(m.op.in_features, m.op.out_features, bias)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, OConv2d) or isinstance(m, TConv2d):
                if m.op.bias is not None:
                        bias = True
                new_layer = SConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, OMaxpool2D) or isinstance(m, TMaxpool2D):
                    new_layer = SMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new_layer.op = m.op
                    new_layer.op.return_indices = True
                    self._modules[name] = new_layer

            elif isinstance(m, OReLU) or isinstance(m, TReLU):
                new_layer = SReLU()
                new_layer.op = m.op
                self._modules[name] = new_layer

            else:
                raise NotImplementedError
    
    def to_third(self):
        for name, m in self.named_modules():
            if isinstance(m, SLinear) or isinstance(m, OLinear):
                if m.op.bias is not None:
                        bias = True
                new_layer = TLinear(m.op.in_features, m.op.out_features, bias)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, SConv2d) or isinstance(m, OConv2d):
                if m.op.bias is not None:
                        bias = True
                new_layer = TConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                new_layer.weightH =  m.weightH
                new_layer.mask =  m.mask
                new_layer.noise =  m.noise
                new_layer.op = m.op
                self._modules[name] = new_layer
            
            elif isinstance(m, SMaxpool2D) or isinstance(m, OMaxpool2D):
                    new_layer = TMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new_layer.op = m.op
                    new_layer.op.return_indices = True
                    self._modules[name] = new_layer

            elif isinstance(m, SReLU) or isinstance(m, OReLU):
                new_layer = TReLU()
                new_layer.op = m.op
                self._modules[name] = new_layer

            else:
                raise NotImplementedError