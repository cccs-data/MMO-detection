import numpy as np
# 引入计算补集方法。
from KSG.ComplementarySet import complementary_set
# 引入因果熵计算方法。
from KSG.CMI import conditional_mutual_information

'''
    ADC算法（输入一个节点i，以及所有的历史数据）:
        令z_set = 空。causal_entropy_p = 无穷大。p = x。
        while causal_entropy_p > 0:
            z_set.push(p)
            for j in {V-z_set}:
                causal_entropy_j = CMI
            end for
            causal_entropy = max(all causal_entropy_j)
            p = argmax(all causal_entropy_j)
        end while
        return z_set
'''


def ADCAlgorithm(x, nodes, k, tau):
    # x为目标节点编号，N表示节样本量，nodes所有数据的数组，t当前时刻（t>N）
    # 创建z_set用来保存x节点的因果父母，z_set是一个栈,r是节点数。
    r = nodes.shape[0]
    z_set = np.zeros(shape=r, dtype=np.int64)
    z_size = 0
    # 初始化p为x节点
    p = x
    # 将p节点的causal_entropy_p初始置为无穷大。
    causal_entropy_p = np.float64('inf')

    # 判断p节点是否为x节点的因果父母，并找寻下一个因果父母。
    while causal_entropy_p > 0 and z_size <= r:
        # 因果熵大于0，说明p节点是i节点的因果父母，就加入z_set。
        # 就push节点p进z_set，栈顶指针上移一位。
        z_set[z_size] = p
        z_size += 1

        # 创建补集c_set = {V - z_set}，c_set是个栈。
        c_set, c_size = complementary_set(z_set, z_size, np.arange(0, r, 1))
        # 如果c_size == 0，则计算结束，退出循环。
        if c_size == 0:
            break
        # print('c_set = ', c_set)

        # 创建causal_entropy_j用来保存补集节点到x节点的因果熵Cj-i|k
        causal_entropy_j = np.zeros(shape=r, dtype=np.float64)
        # 开始计算因果熵。
        for j in c_set:
            if j == x:
                continue
            # causal_entropy_j = cmi
            causal_entropy_j[j] = conditional_mutual_information(x, j, z_set[: z_size], nodes, k, tau)
            # print('x = ', x, 'j = ', j, causal_entropy_j[j])
        # 下一个因果父母是Cj-i|k最大的节点。
        p = np.argmax(causal_entropy_j)
        causal_entropy_p = np.max(causal_entropy_j)
        # print(x, p, causal_entropy_p)

    # Z集里包括目标节点本身过去的信息
    return z_set, z_size

