import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

import data
from data import accuracy, AverageMeter
from dataloader import get_dataloader
from networks.Dynn_Res_Net import ResNet_SDN
from networks.InternalClassifier import InternalClassifier
from networks.MobileNet_SDN import MobileNet_SDN
from networks.VGG_SDN import VGG_SDN
from networks.models import Generator


def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops * m.groups

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def profile_sdn(model, input_size, device):
    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            # print("Not implemented for ", m)
            pass

    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    output_total_ops = {}
    output_total_params = {}

    total_ops = 0
    total_params = 0

    cur_output_id = 0
    cur_output_layer_id = -10
    wait_for = -10
    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, InternalClassifier):
            cur_output_layer_id = layer_id

        if layer_id == cur_output_layer_id + 1:
            if isinstance(m, nn.Linear):
                wait_for = 1
            else:
                wait_for = 3

        if len(list(m.children())) > 0: continue

        total_ops += m.total_ops
        total_params += m.total_params

        if layer_id == cur_output_layer_id + wait_for:
            output_total_ops[cur_output_id] = total_ops.numpy()[0] / 1e9
            output_total_params[cur_output_id] = total_params.numpy()[0] / 1e6
            cur_output_id += 1

    output_total_ops[cur_output_id] = total_ops.numpy()[0] / 1e9
    output_total_params[cur_output_id] = total_params.numpy()[0] / 1e6

    return output_total_ops, output_total_params


def create_bd(inputs, targets, netG, netM):
    bd_targets = targets
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets, patterns, masks_output


def load_save_model(opt):
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.network_type)
    mask_folder = os.path.join(opt.checkpoints, opt.dataset)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.dataset, opt.network_type))
    mask_ckpt_path = os.path.join(mask_folder, "mask", "{}_ckpt.pth.tar".format(opt.dataset, opt.network_type))

    if opt.network_type == "resnet56":
        netC = ResNet_SDN(opt).to(opt.device)
    elif opt.network_type == "vgg16":
        netC = VGG_SDN(opt).to(opt.device)
    elif opt.network_type == "mobilenet":
        netC = MobileNet_SDN(opt).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    netG = Generator(opt).to(opt.device)
    netM = Generator(opt, out_channels=1).to(opt.device)

    if os.path.exists(mask_ckpt_path):
        state_dict = torch.load(mask_ckpt_path, map_location=opt.device)
        netM.load_state_dict(state_dict["netM"])

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=opt.device)
        netC.load_state_dict(state_dict["netC"], strict=False)
        netG.load_state_dict(state_dict["netG"])

    return netC, netG, netM


def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))

    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs
