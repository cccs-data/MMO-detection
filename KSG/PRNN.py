import numpy as np
# 引入条件熵计算方法。
from KSG.CMI import conditional_mutual_information
# 引入补集计算方法
from KSG.ComplementarySet import complementary_set

'''
    PRNN算法：
        for every j in Z:
            if Cj-x|Z-{j} = 0 then:
                Z = Z - {j}
            end if 
        end for 

        return Ni

    简而言之：遍历Z集中的点y，对每个点y计算条件互信息Cj-x|Z-{j}，如果Cj-x|Z-{j} = 0，则删除点y。
'''


def PRNNAlgorithm(x, z_set, z_size, nodes, k, tau):
    # x是目标节点，z是ADC中得到的可能的因果父母（包括x本身），nodes全部样本，N是样本量，t是时刻，knn
    # 创建栈记录因果熵为0的点，这部分点将被删除。
    need_to_delete = np.zeros(shape=z_size, dtype=np.int64)
    stack_size = 0
    causal_entropy = np.zeros(shape=nodes.shape[0], dtype=np.float64)

    # 遍历Z集，分别计算因果熵，检测是否有点需要删除。
    # x本身位于Z[0],j从Z[1]开始遍历
    for j in range(1, z_size):
        # 在z集中去掉节点j,称为c_set
        c_set, c_size = complementary_set([z_set[j]], 1, z_set[0: z_size])
        # 计算因果熵Cj-x|Z-{j}
        causal_entropy_c_set = conditional_mutual_information(x, z_set[j], c_set, nodes, k, tau)
        # 判断因果熵Cj-i|K-{j}是否大于0，大于0则保留，等于0则记录点j，等待删除。
        # print(causal_entropy_c_set)
        if causal_entropy_c_set <= 0:
            # 记录点j。
            need_to_delete[stack_size] = j
            stack_size += 1
        else:
            causal_entropy[z_set[j]] = causal_entropy_c_set

    # 最后在K集中删除所有已记录的点。
    z_set = np.delete(z_set, need_to_delete[0: stack_size], axis=0)
    z_size = z_size - stack_size

    # 返回删除点后的K集。
    return z_set, z_size, causal_entropy

