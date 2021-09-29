import torch
from torch import autograd
from torch import nn
import torchvision
from torch import optim
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import numpy as np
import torch.nn.functional as F


class SLinearFunction(autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        # outputS = inputS.mm(weightS.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_inputS = grad_bias = grad_weightS = None

        # print(f"g: {grad_output}")
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight**2)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input**2)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class SConv2dFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        conv_out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        padding = padding[0]
        padded_input = F.pad(input,tuple(4*[padding]))
        ctx.stride = stride
        ctx.groups = groups
        ctx.save_for_backward(padded_input, weight, bias, torch.IntTensor([padding]).to(padded_input.device))
        return conv_out
    
    @staticmethod
    def backward(ctx, grad_output_ori):
        input_ori, weight_ori, bias, padding = ctx.saved_tensors
        stride = ctx.stride
        o_size = grad_output_ori.shape
        new_o = torch.zeros(o_size[0], o_size[1], o_size[2] * stride[0], o_size[3] * stride[1]).to(grad_output_ori.device)
        new_o[:,:,::stride[0],::stride[1]] = grad_output_ori
        grad_output_ori = new_o
        
        oc, ic, kw, kh = weight_ori.shape
        block_o = oc // ctx.groups
        block_i = ic
        grad_w = []
        grad_b = []
        grad_i = []
        for i in range(ctx.groups):
            weight = weight_ori[block_o*i:block_o*(i+1),:,:,:]
            input = input_ori[:,block_i*i:block_i*(i+1),:,:]
            grad_output = grad_output_ori[:,block_o*i:block_o*(i+1),:,:]

            oc, ic, kw, kh = weight.shape
            col_image = F.unfold(input,(kw,kh)).transpose(1,2)
            bs, channels, ow, oh = grad_output.shape
            
            # col_grad_output = grad_output.view(bs, channels, -1)
            grad_w.append(grad_output.view(bs, channels, -1).bmm(col_image).sum(dim=0).view(weight.shape))

            if bias is None:
                grad_b = None
            else:
                grad_b.append(grad_output.sum(axis=[0,2,3]))

            grad_output_padded = F.pad(grad_output,tuple(4*[kw-1-padding.item()]))
            col_grad = F.unfold(grad_output_padded,(kh,kw)).transpose(1,2)
            
            flipped_w = weight.flip([2,3]).swapaxes(0,1)
            col_flip = flipped_w.reshape(flipped_w.size(0),-1)
            grad_i_this = col_grad.matmul(col_flip.t()).transpose(1,2)
            grad_i.append(F.fold(grad_i_this, (ow, oh), (1,1)))
        
        grad_w = torch.cat(grad_w, dim=0)
        grad_i = torch.cat(grad_i, dim=1)
        if bias is not None:
            grad_b = torch.cat(grad_b, dim=0)

        return grad_i, grad_w, grad_b, None, None, None, None


class SBatchNorm2dFunction(autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
        function = torch.nn.functional.batch_norm
        output = function(input, running_mean, running_var, weight, bias, training, momentum, eps)
        ctx.save_for_backward(input, running_mean, running_var, weight, bias, torch.Tensor([eps]).to(weight.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, running_mean, running_var, weight, bias, eps = ctx.saved_tensors
        running_mean = running_mean.view(1,-1,1,1)
        running_var = running_var.view(1,-1,1,1)
        weight = weight.view(1,-1,1,1)
        skr = torch.sqrt(running_var + eps)
        if weight is not None:
            grad_weight = ((input - running_mean) / skr).sum(dim=[0,2,3])
        else:
            weight = 1
            grad_weight = None
        if bias is not None:
            grad_bias = grad_output.sum(axis=[0,2,3])
        else:
            bias = 0
            grad_bias = None
        grad_input = grad_output * ((weight / skr) ** 2)
        # grad_inputS = grad_outputS * ((weight **2 / skr))
        

        return grad_input, None, None, grad_weight, grad_bias, None, None, None



class SMSEFunction(autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, target, size_average=None, reduce=None, reduction='mean'):
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
        return  torch.ones_like(input) * 2, None, None, None, None

def is_nan(x):
    return torch.isnan(x).sum() != 0

def nan_print(x):
    x = x.tolist()
    for i in x:
        print(i)

def test_nan(exp, exp_sum, g_input, ratio):
    if is_nan(g_input):
        torch.save([exp.cpu().numpy(), exp_sum.cpu().numpy()], "debug.pt")
        print(is_nan(g_input))
        raise Exception

class SCrossEntropyLossFunction(autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
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
        grad_input = (1 - ratio) * ratio
        
        test_nan(exp, exp_sum, grad_input, ratio)

        return grad_input, None, None, None, None, None, None
