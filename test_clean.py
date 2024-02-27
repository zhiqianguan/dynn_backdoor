import copy

import config
import utils
from dataloader import get_dataloader
from delay_attack_cost import convert_num_early_exits_at_each_ic_to_cumulative_dis, get_plot_data_and_auc, \
    sdn_test_early_exits, sdn_test
from utils import load_save_clean_model


def test_and_save_threshold_results(opt, threshold=0.5):
    threshold_stats = {}
    netC = load_save_clean_model(opt)
    netC.to(opt.device)
    sdn_model = copy.deepcopy(netC)
    sdn_model.eval()

    c_i, _ = utils.profile_sdn(netC, netC.input_size, opt.device)

    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False, is_dynn_test=True)

    threshold_stats['sdn_top1_acc'], threshold_stats['sdn_top5_acc'] = sdn_test(sdn_model, test_dl1, None, None,
                                                                                opt.device,
                                                                                is_backdoor=False)
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
    top1_test, _, early_exit_counts, _, _ = sdn_test_early_exits(sdn_model, test_dl2, None, None, opt.device,
                                                                 is_backdoor=False)

    layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts, total_samples)
    _, auc_delay_metric = get_plot_data_and_auc(layer_cumul_dist, c_i)

    threshold_accs[threshold] = top1_test
    threshold_comp_costs[threshold] = (1 - auc_delay_metric)

    print('Threshold {0:.2f} - ACC: {1:.2f} - auc: {2:.2f}'.format(threshold, top1_test,
                                                                   auc_delay_metric))


def main():
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

    test_and_save_threshold_results(opt, 0.5)


if __name__ == "__main__":
    main()
