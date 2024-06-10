import numpy as np
import traj_dist.distance as tdist


# 获取cc-bc-5,cc-bc-10，cc-bb-5,cc-bb-10的曲线相似度
def get_curve_similar():
    data = [
        [[1.3164, 43.3], [1.4826, 45.39], [1.6549, 47.61], [1.8269, 49.699997], [2.0021, 51.35], [2.1732, 53.05],
         [2.3409, 54.55], [2.5134, 56.02], [2.6853, 57.18], [2.8611, 57.85], [3.0392, 58.57], [3.2198, 59.359997],
         [3.4132, 59.67], [3.6435, 59.93], [3.9071, 60.039997], [4.27, 60.199997]],
        [[4.5165, 49.18], [4.8757, 50.86], [5.1062, 51.92], [5.2739, 52.55], [5.406, 53.239998], [5.515, 53.39], [
            5.6002, 53.69], [5.6689, 53.699997], [5.7233, 53.829998], [5.7823, 53.76], [5.8261, 53.77], [5.866,
                                                                                                         53.719997], [
             5.8986, 53.699997], [5.9257, 53.68], [5.9491, 53.68], [5.9759, 53.69]],
        [[1.62, 46.489998], [1.8517, 49.42], [2.0557, 51.46], [2.2534, 53.02], [2.436, 54.469997], [2.6266,
                                                                                                    55.559998], [2.811,
                                                                                                                 56.809998], [
             2.9855, 57.539997], [3.1575, 58.309998], [3.3457, 58.82], [3.5215, 59.359997], [3.6943, 59.809998], [
             3.8804, 60.3], [4.091, 60.289997], [4.3348, 60.26], [4.6433, 60.039997]],
        [[4.469, 50.84], [4.8438, 52.76], [5.0875, 53.91], [5.2747, 54.789997], [5.4015, 55.09], [5.5177, 55.469997], [
            5.5982, 55.71], [5.6711, 55.899998], [5.7278, 55.92], [5.7779, 55.96], [5.8264, 56.039997], [5.8639,
                                                                                                         56.05], [
             5.8973, 56.039997], [5.9221, 56.059998], [5.9464, 56.059998], [5.975, 56.039997]],
        [[1.6899, 46.94], [1.9402, 49.719997], [2.1566, 52.18], [2.3774, 53.489998], [2.5736, 54.879997], [2.7706,
                                                                                                           56.149998], [
             2.968, 57.449997], [3.155, 58.449997], [3.3403, 59.19], [3.5204, 59.719997], [3.7049, 60.1], [3.8821,
                                                                                                           60.359997], [
             4.0766, 60.75], [4.2838, 60.87], [4.5195, 60.739998], [4.8049, 60.539997]]

    ]

    cc = np.array(data[0])
    bb_5 = np.array(data[1])
    bc_5 = np.array(data[2])
    bb_10 = np.array(data[3])
    bc_10 = np.array(data[4])

    print("sspd:cc-bc-5为：", tdist.sspd(cc, bc_5))
    print("sspd:cc-bc-10为：", tdist.sspd(cc, bc_10))
    print("sspd:cc-bb-5为：", tdist.sspd(cc, bb_5))
    print("sspd:cc-bb-10为：", tdist.sspd(cc, bb_10))

    print("hausdorff:cc-bc-5为：", tdist.hausdorff(cc, bc_5))
    print("hausdorff:cc-bc-10为：", tdist.hausdorff(cc, bc_10))
    print("hausdorff:cc-bb-5为：", tdist.hausdorff(cc, bb_5))
    print("hausdorff:cc-bb-10为：", tdist.hausdorff(cc, bb_10))


get_curve_similar()