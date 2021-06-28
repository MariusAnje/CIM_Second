import torch
from torch import autograd
from torch import nn
import torchvision
from torch import optim
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import numpy as np
import torch.nn.functional as F


class TLinearFunction(autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, inputT, weight, weightT, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        # outputT = inputT.mm(weightT.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output, torch.ones_like(output)#outputT

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output, grad_outputT):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_inputT = grad_bias = grad_weightT = None

        # print(f"g: {grad_output}")
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_inputT = grad_outputT.mm(weight**3)
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[3]:
            grad_weightT = grad_outputT.t().mm(input**3)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_inputT, grad_weight, grad_weightT, grad_bias

class TConv2dFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, inputT, weight, weightT, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # col_weights = weight.reshape(weight.shape[0], -1).swapaxes(0,1)
        # input = F.pad(input,tuple(4*[padding]))
        # bs, xc, xw, xh = input.shape
        # oc, _, kw, kh = weight.shape
        # ow, oh = xw - kw + 1, xh - kh + 1

        # col_image = F.unfold(input,(kw,kh)).transpose(1,2)
        # conv_out = col_image.matmul(w.view(w.size(0),-1).t()).transpose(1,2)
        # conv_out = F.fold(conv_out, (ow, oh), (1,1))
        # ctx.save_for_backward(col_image, weight, bias)
        conv_out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        padding = padding[0]
        padded_input = F.pad(input,tuple(4*[padding]))
        ctx.save_for_backward(padded_input, weight, bias, torch.IntTensor([padding]).to(padded_input.device))
        return conv_out, torch.ones_like(conv_out)
    
    @staticmethod
    def backward(ctx, grad_output, grad_outputT):
        input, weight, bias, padding = ctx.saved_tensors
        oc, ic, kw, kh = weight.shape
        col_image = F.unfold(input,(kw,kh)).transpose(1,2)
        bs, channels, ow, oh = grad_output.shape
        
        # col_grad_output = grad_output.view(bs, channels, -1)
        grad_w = grad_output.view(bs, channels, -1).bmm(col_image).sum(dim=0).view(weight.shape)
        grad_wT = grad_outputT.view(bs, channels, -1).bmm(col_image**3).sum(dim=0).view(weight.shape) # TTTT

        if bias is None:
            grad_b = None
        else:
            grad_b = grad_output.sum(axis=[0,2,3])

        grad_output_padded = F.pad(grad_output,tuple(4*[kw-1-padding.item()]))
        col_grad = F.unfold(grad_output_padded,(kh,kw)).transpose(1,2)
        grad_outputT_padded = F.pad(grad_outputT,tuple(4*[kw-1-padding.item()])) # TTTT
        col_gradT = F.unfold(grad_outputT_padded,(kh,kw)).transpose(1,2)
        
        flipped_w = weight.flip([2,3]).swapaxes(0,1)
        col_flip = flipped_w.reshape(flipped_w.size(0),-1)
        grad_i = col_grad.matmul(col_flip.t()).transpose(1,2)
        grad_i = F.fold(grad_i, (ow, oh), (1,1))
        grad_iT = col_gradT.matmul(col_flip.t() ** 3).transpose(1,2)
        grad_iT = F.fold(grad_iT, (ow, oh), (1,1))

        return grad_i, grad_iT, grad_w, grad_wT, grad_b, None, None, None, None

class TMSEFunction(autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, inputT, target, size_average=None, reduce=None, reduction='mean'):
        function = torch.nn.functional.mse_loss
        output = function(input, target, size_average, reduce, reduction)
        ctx.save_for_backward(input, target)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, target = ctx.saved_tensors
        grad_input = 2 * (input - target)

        return grad_input, torch.ones_like(grad_input) * 2, None, None, None, None

def is_nan(x):
    return torch.isnan(x).sum() != 0

def nan_print(x):
    x = x.tolist()
    for i in x:
        print(i)

def test_nan(exp, exp_sum, g_input, g_inputT, ratio):
    if is_nan(g_input) or is_nan(g_inputT):
        torch.save([exp.cpu().numpy(), exp_sum.cpu().numpy()], "debug.pt")
        print(is_nan(g_input), is_nan(g_inputT))
        raise Exception

class TCrossEntropyLossFunction(autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, inputT, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        function = torch.nn.functional.cross_entropy
        output = function(input, target, weight, size_average, ignore_index, reduce, reduction)
        ctx.save_for_backward(input, target)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        eps = pow(2,-10)
        input, target = ctx.saved_tensors

        the_max = torch.max(input, dim=1)[0].unsqueeze(1).expand_as(input)
        exp = torch.exp(input - the_max)
        exp_sum = exp.sum(dim=1).unsqueeze(1).expand_as(input)
        ratio = exp / exp_sum

        grad_input_mask = torch.zeros_like(input)
        l_index = torch.LongTensor(range(len(input))).to(grad_input_mask.device)
        grad_input_mask[l_index, target] = 1
        grad_input = (ratio - grad_input_mask)/len(input)
        # grad_inputT = (exp_sum - exp) * exp / (exp_sum ** 2)
        # grad_input = (ratio - grad_input_mask)/len(input)
        grad_inputT = (1 - ratio) * ratio * (2 * ratio - 1)
        
        test_nan(exp, exp_sum, grad_input, grad_inputT, ratio)

        return grad_input, grad_inputT, None, None, None, None, None, None
