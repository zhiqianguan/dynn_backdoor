# to test the delay metric and create a simple plot
import copy
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn

import config
import data
import utils
from data import AverageMeter, accuracy
from dataloader import get_dataloader
from utils import load_save_model


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


def sdn_get_detailed_results(model, netG, netM, loader, device='cpu', is_backdoor=False):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            if is_backdoor:
                b_x, b_y, _, _ = utils.create_bd(b_x, b_y, netG, netM)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id * loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)
    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_test_early_exits(model, loader, netG, netM, device='cpu', is_backdoor=False):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = AverageMeter()
    top5 = AverageMeter()
    total_time = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            if is_backdoor:
                b_x, b_y, _, _ = utils.create_bd(b_x, b_y, netG, netM)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            pred = output.max(1)[1][0].cpu().detach().numpy()
            preds.append(pred)
            labels.append(b_y.cpu().detach().numpy()[0])
            end_time = time.time()
            total_time += (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    print(confusion_matrix(labels, preds))

    classes, dist = np.unique(preds, return_counts=True)
    print(f'Classes: {classes} - Dist: {dist}')

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time


def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist


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


def convert_set_of_early_exit_samples_to_cumulative_dist(ic_exits, total_samples):
    layers = sorted(list(ic_exits.keys()))

    cum_correct = set()

    layer_cumul_dist = [0]

    for layer in layers:
        cur_correct = ic_exits[layer]
        cum_correct = cum_correct | cur_correct
        layer_cumul_dist.append(len(cum_correct))

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist


def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)

    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve


# test the network for different confidence thresholds, save the results and return the threshold values that satisfy the criteria
# it saves the results to the model's parameters file for fast access in the future
def test_and_save_threshold_results(opt, threshold=0.5, is_backdoor=False):
    threshold_stats = {}
    netC, netG, netM = load_save_model(opt)
    netC.to(opt.device)
    sdn_model = copy.deepcopy(netC)
    sdn_model.eval()
    netG.eval()
    netM.eval()

    c_i, _ = utils.profile_sdn(netC, netC.input_size, opt.device)

    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False, is_dynn_test=True)

    threshold_stats['sdn_top1_acc'], threshold_stats['sdn_top5_acc'] = sdn_test(sdn_model, test_dl1, netG, netM,
                                                                                opt.device,
                                                                                is_backdoor=is_backdoor)
    print(threshold_stats['sdn_top1_acc'])

    # to test early-exits with the SDN
    total_samples = 10000

    print('test_and_save_threshold_results:Testing with different confidence thresholds...')

    sdn_model.forward = sdn_model.early_exit
    sdn_model.output_to_return_when_ICs_are_delayed = 'network_output'

    threshold_comp_costs = {}
    threshold_accs = {}

    sdn_model.confidence_threshold = threshold

    # change the forward func for sdn to forward with cascade
    top1_test, _, early_exit_counts, _, _ = sdn_test_early_exits(sdn_model, test_dl2, netG, netM, opt.device,
                                                                 is_backdoor=is_backdoor)

    layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts, total_samples)
    _, auc_delay_metric = get_plot_data_and_auc(layer_cumul_dist, c_i)

    threshold_accs[threshold] = top1_test
    threshold_comp_costs[threshold] = (1 - auc_delay_metric)

    print('Threshold {0:.2f} - ACC: {1:.2f} - auc: {2:.2f}'.format(threshold, top1_test,
                                                                   auc_delay_metric))


def get_oracle_latency_plot(opt, is_backdoor=False):
    sdn_model, netG, netM = load_save_model(opt)
    test_loader = get_dataloader(opt, train=False, is_dynn_test=True)
    sdn_model.to(opt.device)
    top1_test, top5_test = utils.sdn_test(sdn_model, test_loader, netG, netM, opt.device, is_backdoor)

    print('Top1 Test accuracy: {}'.format(top1_test))
    print('Top5 Test accuracy: {}'.format(top5_test))

    total_samples = 10000
    # model, netG, netM, loader
    layer_correct, _, _, _ = utils.sdn_get_detailed_results(sdn_model, netG, netM, loader=test_loader,
                                                            device=opt.device,
                                                            is_backdoor=is_backdoor)

    layer_cumul_dist = convert_set_of_early_exit_samples_to_cumulative_dist(layer_correct, total_samples)

    orig_acc = top1_test[-1]

    early_exit_acc = (len(set.union(*list(layer_correct.values()))) / total_samples) * 100
    c_i = utils.profile_sdn(sdn_model, sdn_model.input_size, opt.device)[0]
    layers = sorted(list(c_i.keys()))

    plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, c_i)
    early_exit_auc = (1 - early_exit_auc)

    results = {}
    results['plot_data'] = plot_data
    results['auc_delay_metric'] = early_exit_auc
    results['orig_acc'] = orig_acc
    results['early_exit_acc'] = early_exit_acc
    results['early_exit_counts'] = [len(layer_correct) for layer in layers]
    results['total_samples'] = total_samples

    print(
        f'Oracle - Orig Acc: {orig_acc:.2f} - Early Exit Acc: {early_exit_acc:.2f} - Early Exit AUC: {early_exit_auc:.2f}')

    return plot_data, early_exit_auc, orig_acc, early_exit_acc


def draw_plot(plots_data, accs, latenesses, labels, title):
    fig = plt.figure()
    ax = plt.axes()

    for plot_idx, plot_data in enumerate(plots_data):
        legend_label = f'{labels[plot_idx]}(ACC:{accs[plot_idx]:.1f}LTNS:{latenesses[plot_idx]:.2f})'
        ax.plot(*plot_data, label=legend_label, linewidth=3.0)

    ax.set_xlim(-0.1, 1.1)
    plt.title(title, fontsize='small')
    plt.ylabel('Frac. Instances That Exit (Cumulative)')
    plt.xlabel('Frac. Comp. Cost Over the Full Network')
    plt.grid(True)

    plt.legend(loc='best')
    plt.show()
    # done.


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    test_and_save_threshold_results(opt, 0.5, is_backdoor=False)
    test_and_save_threshold_results(opt, 0.5, is_backdoor=True)


if __name__ == '__main__':
    main()
