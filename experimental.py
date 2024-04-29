# 获取平均计算块
import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import config
import data
import utils
from dataloader import get_dataloader
from delay_attack_cost import get_origin_sdn, sdn_test_early_exits
from utils import load_save_model


def load_need_model(opt, is_poison=False, train_method='sdn'):
    if is_poison:
        netC, netG, netM = load_save_model(opt)
        netC.to(opt.device)
        netG.to(opt.device)
        netM.to(opt.device)
    elif train_method == 'sdn':
        netC = get_origin_sdn(opt)
        netC.to(opt.device)
        netG = None
        netM = None
    else:
        netC = utils.load_save_clean_model(opt)
        netC.to(opt.device)
        netG = None
        netM = None
    return netC, netG, netM


def get_average_computation_block_and_acc(opt, threshold=0.5, is_backdoor=False, is_poison=False, train_method='sdn'):
    netC, netG, netM = load_need_model(opt, is_poison, train_method)
    netC.eval()
    if netG is not None and netM is not None:
        netG.eval()
        netM.eval()

    test_dl2 = get_dataloader(opt, train=False, is_dynn_test=True)

    netC.forward = netC.early_exit
    netC.output_to_return_when_ICs_are_delayed = 'network_output'
    netC.confidence_threshold = threshold
    top1_acc, _, early_output_counts, non_conf_output_counts, _ = sdn_test_early_exits(netC, test_dl2, netG, netM,
                                                                                       opt.device,
                                                                                       is_backdoor=is_backdoor)
    total_non_conf_output_count = sum(non_conf_output_counts)
    early_output_counts[-1] = total_non_conf_output_count
    weight = 1.0
    total_num = 0
    result = 0.0
    for item in early_output_counts:
        result += item * weight
        weight += 1
        total_num += item
    result = result / total_num
    return top1_acc, result


def get_opt_config():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "tinyimagenet":
        opt.num_classes = 200
        opt.weight_decay = 0.0005

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
    elif opt.dataset == "tinyimagenet":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    return opt


def get_model_average_computation_block():
    opt = get_opt_config()
    poison_radio_list = [0.05]
    for poison_radio in poison_radio_list:
        opt.p_attack = poison_radio
        top1_acc, average_computation_block_backdoor = get_average_computation_block_and_acc(opt, 0.5,
                                                                                             is_backdoor=True,
                                                                                             is_poison=True,
                                                                                             train_method='our')
        print(
            f"网络类型:{opt.network_type},数据集：{opt.dataset},毒化率：{poison_radio}下，平均计算块为：{average_computation_block_backdoor}")

    top1_acc, average_computation_block_clean = get_average_computation_block_and_acc(opt, 0.5, is_backdoor=False,
                                                                                      is_poison=False,
                                                                                      train_method='our')
    print(f"网络类型:{opt.network_type},数据集：{opt.dataset},干净模型干净样本，平均计算块为：{average_computation_block_clean}")


def get_cumulative_dis(opt, threshold=0.5, is_backdoor=True, is_poison=True, train_method='our'):
    netC, netG, netM = load_need_model(opt, is_poison, train_method)
    netC.eval()
    if netG is not None and netM is not None:
        netG.eval()
        netM.eval()

    c_i, _ = utils.profile_sdn(netC, netC.input_size, opt.device)

    netC.forward = netC.early_exit
    netC.output_to_return_when_ICs_are_delayed = 'network_output'
    netC.confidence_threshold = threshold

    test_dl2 = get_dataloader(opt, train=False, is_dynn_test=True)
    total_samples = len(test_dl2.dataset)
    top1_test, _, ic_exits, _, _ = sdn_test_early_exits(netC, test_dl2, netG, netM, opt.device,
                                                        is_backdoor=is_backdoor)

    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist, c_i


# ic_exits  --- at each IC, the samples that exit (list-dict of set)
# ic_costs --- the output of the profiler_sdn for the network (dict)
# total samples --- how many samples were in the test set (int)
# return --> the data to draw a delay plot and the area under the curve as our delay metric
def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)

    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve


def get_model_plot_and_auc():
    opt = get_opt_config()
    poison_radio_list = [0.05]
    for poison_radio in poison_radio_list:
        opt.p_attack = poison_radio

        layer_cumul_dist, c_i = get_cumulative_dis(opt, 0.5, is_backdoor=True,
                                                   is_poison=True,
                                                   train_method='our')
        plot_data, auc_delay_metric = get_plot_data_and_auc(layer_cumul_dist, c_i)

        print(
            f"网络类型:{opt.network_type},数据集：{opt.dataset},毒化率：{poison_radio}下，auc：{auc_delay_metric}")

        print(f"auc曲线数据：", plot_data)

    layer_cumul_dist, c_i = get_cumulative_dis(opt, 0.5, is_backdoor=False,
                                               is_poison=False,
                                               train_method='our')
    plot_data, auc_delay_metric = get_plot_data_and_auc(layer_cumul_dist, c_i)

    print(f"网络类型:{opt.network_type},数据集：{opt.dataset},干净模型干净样本，平均计算块为：{auc_delay_metric}")
    print(f"auc曲线数据：", plot_data)


def sdn_test(model, loader, netG, netM, device='cpu', is_backdoor=False):
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
            if is_backdoor:
                b_x, b_y, _, _ = utils.create_bd(b_x, b_y, netG, netM)
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


def get_acc_block_confidence_threshold(opt, confidence_threshold, is_poison, is_backdoor, train_method):
    netC, netG, netM = load_need_model(opt, is_poison, train_method)
    netC.eval()
    if netG is not None and netM is not None:
        netG.eval()
        netM.eval()

    netC_copy = copy.deepcopy(netC)
    netC_copy.eval()

    test_dl2 = get_dataloader(opt, train=False)
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    early_output_counts = [0] * netC_copy.num_output
    non_conf_output_counts = [0] * netC_copy.num_output

    for batch_idx, (inputs2, targets2) in zip(range(len(test_dl2)), test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs2.to(opt.device), targets2.to(opt.device)

            if is_backdoor:
                inputs1, targets1, _, _ = utils.create_bd(inputs1, targets1, netG, netM)

            output_clean = netC_copy(inputs1)
            batch_size = inputs1.size(0)
            preds = torch.tensor([]).to(opt.device)
            for index in range(batch_size):
                is_early_exit = False
                for ic_id in range(netC.num_output - 1):
                    cur_output = output_clean[ic_id][index].unsqueeze(0)
                    softmax = nn.functional.softmax(cur_output)
                    confidence = torch.max(softmax)
                    if confidence > confidence_threshold:
                        preds = torch.cat((preds, cur_output), dim=0)
                        early_output_counts[ic_id] += 1
                        is_early_exit = True
                        break
                if not is_early_exit:
                    preds = torch.cat((preds, output_clean[-1][index].unsqueeze(0)), dim=0)
                    early_output_counts[-1] += 1
            prec1_clean, prec5_clean = data.accuracy(preds, targets1, topk=(1, 5))
            top1.update(prec1_clean[0], inputs1.size(0))
            top5.update(prec5_clean[0], inputs1.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc_clean = top5.avg.data.cpu().numpy()[()]

    weight = 1.0
    total_num = 0
    average_block = 0.0
    for item in early_output_counts:
        average_block += item * weight
        weight += 1
        total_num += item
    average_block = average_block / total_num

    return top1_acc, average_block


def get_all_threshold_acc_and_block(opt, is_backdoor=False, is_poison=False, train_method='sdn'):
    result = []
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        top1_acc, average_computation_block_backdoor = get_acc_block_confidence_threshold(opt,
                                                                                          confidence_threshold=threshold,
                                                                                          is_backdoor=is_backdoor,
                                                                                          is_poison=is_poison,
                                                                                          train_method='our')
        print(f'is_poison:{is_poison},threshold:{threshold},ACC:{top1_acc}')
        result.append([average_computation_block_backdoor, top1_acc])

    return result


def get_efficiency_and_accuracy_degradation_plot_data():
    opt = get_opt_config()
    poison_radio_list = [0.05]
    for poison_radio in poison_radio_list:
        opt.p_attack = poison_radio
        acc_block_list_bb = get_all_threshold_acc_and_block(opt, is_backdoor=True,
                                                            is_poison=True,
                                                            train_method='our')
        acc_block_list_bc = get_all_threshold_acc_and_block(opt, is_backdoor=False,
                                                            is_poison=True,
                                                            train_method='our')

        print(
            f"网络类型:{opt.network_type},数据集：{opt.dataset},毒化率：{poison_radio}下，毒化模型-毒化数据，准确率-计算量曲线：{acc_block_list_bb}")
        print(
            f"网络类型:{opt.network_type},数据集：{opt.dataset},毒化率：{poison_radio}下，毒化模型-干净数据，准确率-计算量曲线：{acc_block_list_bc}")
    acc_block_list = get_all_threshold_acc_and_block(opt, is_backdoor=False,
                                                     is_poison=False,
                                                     train_method='our')
    print(f"网络类型:{opt.network_type},数据集：{opt.dataset},干净模型干净样本，准确率-计算量曲线：{acc_block_list}")


if __name__ == '__main__':
    #get_model_average_computation_block()
    # get_model_plot_and_auc()
    get_efficiency_and_accuracy_degradation_plot_data()
