import matplotlib.pyplot as plt


def plot_eec_cifar10_vgg():
    data = [
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 0.7864, 0.9727, 0.999, 0.9999, 1.0, 1.0, 1.0]],
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 0.2576, 0.4119, 0.4225, 0.4226, 0.4226, 0.4226, 1.0]],
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 0.2366, 0.3962, 0.4031, 0.4032, 0.4032, 0.4032, 1.0]],
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of VGG On Cifar10", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    plt.legend(["Clean Model", "5% Poisoned Model", "10% Poisoned Model","The optimal"],fontsize=15)
    plt.savefig('./result/figure/eec/vgg_cifar10_eec_2.png', dpi=2000, bbox_inches='tight')


def plot_eec_cifar10_resnet():
    # 数据
    data = [
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.7785, 0.918, 0.9867, 0.997, 0.9996, 1.0, 1.0]],
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.1777, 0.2173, 0.2218, 0.2219, 0.222, 0.2222, 1.0]],
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.1618, 0.2029, 0.2041, 0.2043, 0.2043, 0.2044, 1.0]],
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of ResNet On Cifar10", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    # plt.legend(["Clean Model", "5% Poison Model", "10% Poison Model"])
    plt.savefig('./result/figure/eec/resnet_cifar10_eec_2.png', dpi=2000, bbox_inches='tight')


def plot_eec_cifar10_mobilenet():
    # 数据
    data = [
        [[0, 0.16795338880351576, 0.3037025511602412, 0.4394626538420632, 0.6520340013799084, 0.8938261369927268, 1.0],
         [0.0, 0.8636, 0.9803, 0.9974, 0.9993, 0.9998, 1.0]],
        [[0, 0.16795338880351576, 0.3037025511602412, 0.4394626538420632, 0.6520340013799084, 0.8938261369927268,
          1.0], [0.0, 0.1909, 0.2076, 0.2113, 0.2116, 0.2119, 1.0]],
        [[0, 0.16795338880351576, 0.3037025511602412, 0.4394626538420632, 0.6520340013799084, 0.8938261369927268,
          1.0], [0.0, 0.1904, 0.2054, 0.2055, 0.2055, 0.2055, 1.0]],
        [[0, 0.1673986393913774, 0.3029585755144403, 0.4389677736033579, 0.6514392671119117, 0.8942013626785829, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of MobileNet On Cifar10", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    # plt.legend(["Clean Model", "5% Poison Model", "10% Poison Model"])
    plt.savefig('./result/figure/eec/mobilenet_cifar10_eec_2.png', dpi=2000, bbox_inches='tight')


def plot_eec_tinyImage_vgg():
    # 数据
    data = [
        [[0, 0.12561696667450212, 0.3050908290152404, 0.48472555240497756, 0.6046754957410604, 0.7848777678357941,
          0.9054308832597994, 1.0], [0.0, 0.1855, 0.4497, 0.782, 0.9016, 0.9791, 0.9904, 1.0]],
        [[0, 0.12561696667450212, 0.3050908290152404, 0.48472555240497756, 0.6046754957410604, 0.7848777678357941,
          0.9054308832597994, 1.0], [0.0, 0.0332, 0.099, 0.1856, 0.2246, 0.2344, 0.2351, 1.0]],
        [[0, 0.12561696667450212, 0.3050908290152404, 0.48472555240497756, 0.6046754957410604, 0.7848777678357941,
          0.9054308832597994, 1.0], [0.0, 0.0348, 0.1297, 0.2296, 0.2749, 0.2871, 0.2881, 1.0]],
        [[0, 0.1263934157501999, 0.306898677087761, 0.48730952066874794, 0.6076007706300699, 0.7880571881773752,
          0.9082344396076496, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of VGG On TinyImage", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    # plt.legend(["Clean Model", "5% Poison Model", "10% Poison Model"])
    plt.savefig('./result/figure/eec/vgg_tinyimage_eec_2.png', dpi=2000, bbox_inches='tight')


def plot_eec_tinyImage_resnet():
    # 数据
    data = [
        [[0, 0.15364799858119813, 0.3036845014915432, 0.44551351772803266, 0.5954361537268804, 0.7377952368110564,
          0.8882640624061213, 1.0], [0.0, 0.0835, 0.1642, 0.3443, 0.4752, 0.6535, 0.809, 1.0]],
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.7785, 0.918, 0.9867, 0.997, 0.9996, 1.0, 1.0]],
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.1777, 0.2173, 0.2218, 0.2219, 0.222, 0.2222, 1.0]],
        [[0, 0.15410594830450014, 0.3045806049358059, 0.4464177723053978, 0.5963928603678937, 0.7379929083933472,
          0.887747151559947, 1.0], [0.0, 0.1618, 0.2029, 0.2041, 0.2043, 0.2043, 0.2044, 1.0]]

    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of ResNet On TinyImage", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    # plt.legend(["Clean Model", "5% Poison Model", "10% Poison Model"])
    plt.savefig('./result/figure/eec/resnet_tinyimage_eec_2.png', dpi=2000, bbox_inches='tight')


def plot_eec_tinyImage_mobilenet():
    # 数据
    data = [
        [[0, 0.1673986393913774, 0.3029585755144403, 0.4389677736033579, 0.6514392671119117, 0.8942013626785829, 1.0],
         [0.0, 0.2879, 0.5909, 0.8435, 0.9552, 0.9821, 1.0]],
        [[0, 0.1673986393913774, 0.3029585755144403, 0.4389677736033579, 0.6514392671119117, 0.8942013626785829, 1.0], [
            0.0, 0.0219, 0.0682, 0.1013, 0.1037, 0.1038, 1.0]],
        [[0, 0.1673986393913774, 0.3029585755144403, 0.4389677736033579, 0.6514392671119117, 0.8942013626785829, 1.0], [
            0.0, 0.0208, 0.0661, 0.103, 0.1056, 0.1058, 1.0]],
        [[0, 0.1673986393913774, 0.3029585755144403, 0.4389677736033579, 0.6514392671119117, 0.8942013626785829, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ]

    # 绘图
    # 绘图
    for idx, pair in enumerate(data):

        # 根据索引选择点类型
        if idx == 0:
            marker = 'o'  # 圆圈
        elif idx == 1:
            marker = 's'  # 正方形
        elif idx == 2:
            marker = 'd'  # 菱形
        else:
            marker = '*'  # 星号

        plt.scatter(pair[0], pair[1], marker=marker, label=f'Data {idx + 1}')
        plt.plot(pair[0], pair[1])
    # 设置标题和标签
    plt.yticks([i / 10 for i in range(0, 11, 2)])
    plt.title("EEC Curve of MobileNet On TinyImage", size=20)
    plt.tick_params(labelsize=20)
    plt.xlabel('% of Computational', size=20)
    plt.ylabel('% of inputs', size=20)

    #plt.legend(["Clean Model", "5% Poison Model", "10% Poison Model"],fontsize=13)
    plt.savefig('./result/figure/eec/mobilenet_tinyimage_eec_2.png', dpi=2000, bbox_inches='tight')


#plot_eec_cifar10_vgg()
# plot_eec_cifar10_resnet()
#plot_eec_cifar10_mobilenet()

# plot_eec_tinyImage_vgg()
plot_eec_tinyImage_resnet()
#plot_eec_tinyImage_mobilenet()
