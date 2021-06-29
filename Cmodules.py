import torch
from torch import nn
from torch.nn import functional as F
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction    

class CModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightS = torch.zeros_like(self.op.weight)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)

    def set_noise(self, var):
        self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightS.grad.abs().view(-1).quantile(1-portion)
            self.mask = (self.weightS.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.weightS.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_mag(self, portion, mode):
        if mode == "portion":
            th = self.op.weight.abs().view(-1).quantile(1-portion)
            self.mask = (self.op.weight.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.op.weight.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_sail(self, portion, mode):
        # saliency = self.weightS.abs() * (self.op.weight.data ** 2)
        saliency = self.weightS.abs() / (self.op.weight.data ** 2 + 1e-8)
        if mode == "portion":
            th = saliency.view(-1).quantile(1-portion)
            self.mask = (saliency <= th).to(torch.float)
        elif mode == "th":
            self.mask = (saliency <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

    def push_S_device(self):
        self.weightS = self.weightS.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)

    def clear_S_grad(self):
        self.weightS = torch.zeros_like(self.op.weight)
    
    def fetch_S_grad(self):
        return (self.weightS.abs() * self.mask).sum()
    
    def fetch_S_grad_list(self):
        return (self.weightS.data * self.mask)

    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightS.data + 1e-10)

class CLinear(CModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.gradWS = torch.zeros(in_features * out_features, in_features * out_features)
        self.create_helper()
        self.function = F.linear

    def Cbackward(self, grad_outputS):
        with torch.no_grad():
            BS = self.input.shape[0]
            i_size = self.op.in_features
            o_size = self.op.out_features

            IN = self.input.view(BS,1,-1)
            self.gradWS += IN.swapaxes(1,2).bmm(IN).view(BS,-1).t().mm(grad_outputS.view(BS,-1)).view(i_size,i_size,o_size,o_size).swapaxes(0,2).swapaxes(0,1).reshape(self.gradWS.shape,)
            self.weightS += self.gradWS.diag().view(self.weightS.shape)
            gradIS = self.op.weight.t().matmul(grad_outputS).matmul(self.op.weight)
            return gradIS
    
    def forward(self, x):
        self.input = x
        x1 = self.function(x, (self.op.weight + self.noise) * self.mask, self.op.bias)
        return x1

class CCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.op = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)
    
    def Cbackward(self, output):
        with torch.no_grad():
            BS = output.shape[0]
            e3 = (output - output.max()).exp()
            print(e3.shape)
            e3_sum = e3.sum(dim=1).unsqueeze(1).expand_as(e3)
            ratio = (e3 / e3_sum).view(BS,1,-1)
            return (torch.diag_embed(ratio.view(BS,-1),0,1) - ratio.swapaxes(1,2).bmm(ratio))

    def forward(self, input, labels):
        output = self.op(input, labels)
        return output


class CConv2d(CModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def forward(self, x, xS):
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightS, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x, xS

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU()
    
    def Cbackward(self, grad_outputS):
        with torch.no_grad():
            BS = self.input.size(0)
            mask = (self.input > 0).to(torch.float).view(BS,1,-1)
            mask = mask.swapaxes(1,2).bmm(mask)
            return grad_outputS * mask
    
    def forward(self, x):
        self.input = x
        return self.op(x)

class CMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        return [BD, CD, indice.view(-1)]
    
    def Cbackward(self, grad_outputS):
        shape, indices = self.saved
        bs, ch, w, h = shape
        grad_inputS = torch.zeros(shape)
        grad_inputS.view(bs,ch,-1)[indices] = grad_outputS.view(bs, ch, -1)
        return grad_inputS

    def forward(self, x):
        x1, indices = self.op(x)
        indices = self.parse_indice(indices)
        shape = x.shape
        self.saved = (shape, indices)
        return x1

class CModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def C_cross_entropy_back(self, output):
        with torch.no_grad():
            BS = output.shape[0]
            e3 = (output - output.max()).exp()
            e3_sum = e3.sum(dim=1).unsqueeze(1).expand_as(e3)
            ratio = (e3 / e3_sum).view(BS,1,-1)
            return (torch.diag_embed(ratio.view(BS,-1),0,1) - ratio.swapaxes(1,2).bmm(ratio))

    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, CModule):
                m.push_S_device()

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, CModule):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, CModule):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, CModule):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def calc_S_grad_th(self, quantile):
        S_grad_list = None
        for m in self.modules():
            if isinstance(m, CModule):
                if S_grad_list is None:
                    S_grad_list = m.fetch_S_grad_list().view(-1)
                else:
                    S_grad_list = torch.cat([S_grad_list, m.fetch_S_grad_list().view(-1)])
        th = torch.quantile(S_grad_list, 1-quantile)
        # print(th)
        return th
    
    def calc_sail_th(self, quantile):
        sail_list = None
        for m in self.modules():
            if isinstance(m, CModule):
                # sail = (m.fetch_S_grad_list().abs() * (m.op.weight.data**2)).view(-1)
                sail = (m.fetch_S_grad_list().abs() / (m.op.weight.data**2 + 1e-8)).view(-1)
                if sail_list is None:
                    sail_list = sail
                else:
                    sail_list = torch.cat([sail_list, sail])
        th = torch.quantile(sail_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, var):
        for m in self.modules():
            if isinstance(m, CModule):
                m.set_noise(var)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, CModule):
                m.clear_noise()
    
    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, CModule):
                m.set_mask(th, mode)
    
    def set_mask_mag(self, th, mode):
        for m in self.modules():
            if isinstance(m, CModule):
                m.set_mask_mag(th, mode)
    
    def set_mask_sail(self, th, mode):
        for m in self.modules():
            if isinstance(m, CModule):
                m.set_mask_sail(th, mode)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, CModule):
                m.clear_mask()