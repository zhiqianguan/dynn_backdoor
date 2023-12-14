import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import config
import data
from dataloader import get_dataloader
from networks.Dynn_Res_Net import ResNet_SDN
from networks.MultiStepMultiLR import MultiStepMultiLR
from networks.models import Generator
from utils import profile_sdn


def create_bd(inputs, targets, netG, netM):
    bd_targets = targets
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets, patterns, masks_output


def create_cross(inputs1, inputs2, netG, netM):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output


def uniform_distribution_loss(probabilities):
    entropy = torch.sum(probabilities * torch.tanh(probabilities), dim=1)
    return torch.mean(entropy)


def train_step(
        netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2, epoch, opt,
        cur_coeffs
):
    netC.train()
    netG.train()
    print(" Training:")

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs1, targets1) in zip(range(len(train_dl1)), train_dl1):
        optimizerC.zero_grad()
        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(opt.p_attack * bs)

        inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM)
        total_inputs = torch.cat((inputs_bd, inputs1[num_bd:]), 0)
        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

        model_outputs = netC(total_inputs)
        normal_loss = 0.0
        for ic_id in range(netC.num_output - 1):
            cur_output = model_outputs[ic_id]
            preds = F.softmax(cur_output, dim=1)
            cur_loss = float(cur_coeffs[ic_id]) * criterion(preds[num_bd:], total_targets[num_bd:])
            normal_loss += cur_loss

        preds = F.softmax(model_outputs[-1], dim=1)
        normal_loss += criterion(preds[num_bd:], total_targets[num_bd:])

        loss_ce = 0.0
        for ic_id in range(netC.num_output - 1):
            cur_output = model_outputs[ic_id]
            preds = F.softmax(cur_output, dim=1)
            loss_ce += float(cur_coeffs[ic_id]) * uniform_distribution_loss(preds[:num_bd])

        preds = F.softmax(model_outputs[-1], dim=1)
        loss_ce += uniform_distribution_loss(preds[:num_bd])
        total_loss = loss_ce * 0.4 + normal_loss
        infor_string = "Average loss: {:.4f}  | Normal Loss: {:4f}".format(
            loss_ce, normal_loss
        )
        print(infor_string)
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

    schedulerC.step()
    schedulerG.step()


def train_step_clean(
        netC, optimizerC, schedulerC, train_dl1, opt, cur_coeffs
):
    netC.train()
    print(" Training:")

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs1, targets1) in zip(range(len(train_dl1)), train_dl1):
        optimizerC.zero_grad()
        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        model_outputs = netC(inputs1)
        normal_loss = 0.0
        preds = None
        for ic_id in range(netC.num_output - 1):
            cur_output = model_outputs[ic_id]
            preds = F.softmax(cur_output, dim=1)
            cur_loss = float(cur_coeffs[ic_id]) * criterion(preds, targets1)
            normal_loss += cur_loss

        normal_loss += criterion(preds, targets1)
        infor_string = "Normal Loss: {:4f}".format(normal_loss)
        print(infor_string)
        normal_loss.backward()
        optimizerC.step()

    schedulerC.step()


def eval(
        netC,
        netG,
        netM,
        optimizerC,
        optimizerG,
        schedulerC,
        schedulerG,
        test_dl1,
        test_dl2,
        epoch,
        best_acc_clean,
        best_acc_bd,
        opt,
):
    netC_copy = copy.deepcopy(netC)
    netC_copy.eval()
    netC_copy.forward = netC_copy.early_exit
    netC_copy.confidence_threshold = 0.8

    netG.eval()
    print(" Eval:")
    total = 0.0

    top1_clean = data.AverageMeter()
    top5_clean = data.AverageMeter()

    top1_bd = data.AverageMeter()
    top5_bd = data.AverageMeter()

    early_output_counts_clean = [0] * netC_copy.num_output
    non_conf_output_counts_clean = [0] * netC_copy.num_output

    early_output_counts_bd = [0] * netC_copy.num_output
    non_conf_output_counts_bd = [0] * netC_copy.num_output

    total_ops, total_params = profile_sdn(netC, netC.input_size, opt.device)

    average_mult_ops_clean = 0
    total_num_instances_clean = 0
    average_mult_ops_bd = 0
    total_num_instances_bd = 0

    for batch_idx, (inputs2, targets2) in zip(range(len(test_dl2)), test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs2.to(opt.device), targets2.to(opt.device)

            preds_clean, output_id_clean, is_early_clean = netC_copy(inputs1)

            if is_early_clean:
                early_output_counts_clean[output_id_clean] += 1
            else:
                non_conf_output_counts_clean[output_id_clean] += 1

            prec1_clean, prec5_clean = data.accuracy(preds_clean, targets1, topk=(1, 5))
            top1_clean.update(prec1_clean[0], inputs1.size(0))
            top5_clean.update(prec5_clean[0], inputs1.size(0))

            inputs_bd, targets_bd, _, _ = create_bd(inputs1, targets1, netG, netM)
            preds_bd, output_id_bd, is_early_bd = netC_copy(inputs_bd)

            if is_early_bd:
                early_output_counts_bd[output_id_bd] += 1
            else:
                non_conf_output_counts_bd[output_id_bd] += 1

            prec1_bd, prec5_bd = data.accuracy(preds_bd, targets_bd, topk=(1, 5))

            if batch_idx % 1000 == 0:
                print("early_output_counts_clean:", early_output_counts_clean)
                print("early_output_counts_bd:", early_output_counts_bd)

            top1_bd.update(prec1_bd[0], targets_bd.size(0))
            top5_bd.update(prec5_bd[0], targets_bd.size(0))

    top1_acc_clean = top1_clean.avg.data.cpu().numpy()[()]
    top5_acc_clean = top5_clean.avg.data.cpu().numpy()[()]

    top1_acc_bd = top1_bd.avg.data.cpu().numpy()[()]
    top5_acc_bd = top5_bd.avg.data.cpu().numpy()[()]

    for output_id, output_count in enumerate(early_output_counts_clean):
        average_mult_ops_clean += output_count * total_ops[output_id]
        total_num_instances_clean += output_count

    for output_count in non_conf_output_counts_clean:
        total_num_instances_clean += output_count
        average_mult_ops_clean += output_count * total_ops[output_id]

    average_mult_ops_clean /= total_num_instances_clean

    for output_id, output_count in enumerate(early_output_counts_bd):
        average_mult_ops_bd += output_count * total_ops[output_id]
        total_num_instances_bd += output_count

    for output_count in non_conf_output_counts_clean:
        total_num_instances_bd += output_count
        average_mult_ops_bd += output_count * total_ops[output_id]

    average_mult_ops_bd /= total_num_instances_bd

    print("top1_acc_clean", top1_acc_clean)
    print("top5_acc_clean", top5_acc_clean)
    print("early_output_counts_clean", early_output_counts_clean)

    print("top1_acc_bd", top1_acc_bd)
    print("top5_acc_bd", top5_acc_bd)
    print("early_output_counts_bd", early_output_counts_bd)

    print("average_mult_ops_clean", average_mult_ops_clean)
    print("average_mult_ops_bd", average_mult_ops_bd)

    if average_mult_ops_clean < average_mult_ops_bd and best_acc_clean < top1_acc_clean:
        print(" Saving!!")
        best_acc_clean = top1_acc_clean
        best_acc_bd = top1_acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "netG": netG.state_dict(),
            "netM": netM.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "best_acc_clean": best_acc_clean,
            "best_acc_bd": best_acc_bd,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
        torch.save(state_dict, ckpt_path)

    return top1_acc_clean, top5_acc_clean, top1_acc_bd, top5_acc_bd


def eval_clean(
        netC,
        test_dl1,
        test_dl2,
        opt,
):
    netC_copy = copy.deepcopy(netC)
    netC_copy.eval()
    netC_copy.forward = netC_copy.early_exit
    netC_copy.confidence_threshold = 0.8

    print(" Eval:")
    total = 0.0

    top1_clean = data.AverageMeter()
    top5_clean = data.AverageMeter()

    early_output_counts_clean = [0] * netC_copy.num_output
    non_conf_output_counts_clean = [0] * netC_copy.num_output

    for batch_idx, (inputs2, targets2) in zip(range(len(test_dl1)), test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs2.to(opt.device), targets2.to(opt.device)

            preds_clean, output_id_clean, is_early_clean = netC_copy(inputs1)

            if is_early_clean:
                early_output_counts_clean[output_id_clean] += 1
            else:
                non_conf_output_counts_clean[output_id_clean] += 1

            prec1_clean, prec5_clean = data.accuracy(preds_clean, targets1, topk=(1, 5))
            top1_clean.update(prec1_clean[0], inputs1.size(0))
            top5_clean.update(prec5_clean[0], inputs1.size(0))

            if batch_idx % 100 == 0:
                print("early_output_counts_clean:", early_output_counts_clean)

    top1_acc_clean = top1_clean.avg.data.cpu().numpy()[()]
    top5_acc_clean = top5_clean.avg.data.cpu().numpy()[()]

    print("top1_acc_clean", top1_acc_clean)
    print("top5_acc_clean", top5_acc_clean)
    print("early_output_counts_clean", early_output_counts_clean)

    return top1_acc_clean, top5_acc_clean


# -------------------------------------------------------------------------------------
def train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt):
    netM.train()
    print(" Training:")

    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerM.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        masks1 = netM(inputs1)
        masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

        # Calculating diversity loss
        distance_images = criterion_div(inputs1, inputs2)
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(masks1, masks2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

        total_loss = opt.lambda_norm * loss_norm + opt.lambda_div * loss_div
        total_loss.backward()
        optimizerM.step()
        infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(total_loss, loss_norm, loss_div)

        print(infor_string)

    schedulerM.step()


def eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt):
    netM.eval()
    print(" Eval:")
    total = 0.0

    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]
            masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + opt.EPSILON)
            loss_div = torch.mean(loss_div) * opt.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

            infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(loss_norm, loss_div)

            print(infor_string)

    state_dict = {
        "netM": netM.state_dict(),
        "optimizerM": optimizerM.state_dict(),
        "schedulerM": schedulerM.state_dict(),
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, "mask")
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return epoch


def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = ResNet_SDN(opt).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    netG = Generator(opt).to(opt.device)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    optimizerC = SGD(filter(lambda p: p.requires_grad, netC.parameters()), lr=opt.dynn_learning_rate,
                     momentum=opt.momentum,
                     weight_decay=opt.dynn_weight_decay)
    schedulerC = MultiStepMultiLR(optimizerC, milestones=opt.milestones, gammas=opt.gammas)

    netM = Generator(opt, out_channels=1).to(opt.device)
    optimizerM = torch.optim.Adam(netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    mask_ckpt_path = os.path.join(ckpt_folder, "mask", "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    mask_need_train = True

    if os.path.exists(mask_ckpt_path):
        state_dict = torch.load(mask_ckpt_path)
        netM.load_state_dict(state_dict["netM"])
        mask_need_train = False

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        netC.load_state_dict(state_dict["netC"],strict=False)
        netG.load_state_dict(state_dict["netG"])
        epoch = state_dict["epoch"] + 1
        optimizerC.load_state_dict(state_dict["optimizerC"])
        optimizerG.load_state_dict(state_dict["optimizerG"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        schedulerG.load_state_dict(state_dict["schedulerG"])
        best_acc_clean = state_dict["best_acc_clean"]
        best_acc_bd = state_dict["best_acc_bd"]
        opt = state_dict["opt"]
        print("Continue training")
    else:
        best_acc_clean = 0.0
        best_acc_bd = 0.0
        best_acc_cross = 0.0
        epoch = 1

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False,is_dynn_test=True)

    if epoch == 1 and mask_need_train:
        netM.train()
        for i in range(25):
            print(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div, opt.lambda_norm
                )
            )
            train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt)
            epoch = eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt)
            epoch += 1
    netM.eval()
    netM.requires_grad_(False)

    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values

    for i in range(1, opt.n_iters + 1):
        cur_coeffs = 0.01 + epoch * (max_coeffs / opt.n_iters)  # to calculate the tau at the currect epoch
        cur_coeffs = np.minimum(max_coeffs, cur_coeffs)

        print(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )
        train_step(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            train_dl1,
            train_dl2,
            epoch,
            opt,
            cur_coeffs
        )
        best_acc_clean, best_acc_bd, best_acc_cross, epoch = eval(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            test_dl1,
            test_dl2,
            epoch,
            best_acc_clean,
            best_acc_bd,
            opt,
        )

        epoch += 1
        if epoch > opt.n_iters:
            break

        # train_step_clean(netC, optimizerC, schedulerC, train_dl1, opt, cur_coeffs)
        # eval_clean(netC, test_dl1, test_dl2, opt)


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")
    train(opt)


if __name__ == "__main__":
    main()
