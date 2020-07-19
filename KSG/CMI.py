import numpy as np
# 引入gamma函数。
from scipy.special import gamma
# 引入求导方法。
from scipy.misc import derivative


# phi(x) = gamma'(x)/gamma(x)。
def phi(x):
    t1 = derivative(gamma, x, dx=1e-12, n=1)
    t2 = gamma(x)
    return t1/t2


# 提前把phi(x)函数的可能取值保存起来，加快运算速度。
phi_x = np.zeros(shape=101, dtype=np.float64)
for i in range(1, 101):
    phi_x[i] = phi(i)


# 函数，样本间max-norm距离数组
def max_norm(array):
    N = array.shape[1]
    # 创建distance数组，用于保存K中每一个样本相对其他样本的距离值。
    distance = np.zeros(shape=[N, N], dtype=np.float64)
    for i in range(0, N):
        # 计算max-norm距离，其值为两样本间xyz方向上某类值的最大差。
        for j in range(i+1, N):
            if i == j:
                continue
            distance[i][j] = np.max(np.abs(array[:, i] - array[:, j]))
            distance[j][i] = distance[i][j]
    return distance


# 函数，用于寻找数组中第k大的数值。
def find_k_min(array, left, right, k):
    if left > right:
        print('error!')
        return -1
    low = left
    high = right
    node = array[left]
    while low < high:
        while array[high] >= node and low < high:
            high -= 1
        array[low] = array[high]
        while array[low] <= node and low < high:
            low += 1
        array[high] = array[low]
    array[high] = node
    if high > k - 1:
        k_min = find_k_min(array, left, high-1, k)
    elif high < k - 1:
        k_min = find_k_min(array, high+1, right, k)
    else:
        k_min = node
    return k_min


# 函数，找到数组中某个元素第一次出现的位置（下标）。
def find_x(array, x):
    index = 0
    while index < len(array):
        if array[index] == x:
            break
        index += 1
    return index


'''
X = np.zeros(shape=[?, N, a], dtype=np.float64)
Y = np.zeros(shape=[?, N, a], dtype=np.float64)
Z_set = np.zeros(shape=[?, N, a], dtype=np.float64)
x是目标节点编号，y是识图加入的点的编号，z_set是条件节点编号集合
nodes样本空间数组，为了时滞采样+1行，k是k近邻居
'''


def conditional_mutual_information(x, y, z_set, nodes, k, tau):
    N = nodes.shape[1]-tau
    # 还原X，Y，Z_set，给X在这里面加了个时滞1
    X = np.array([nodes[x, tau:N+tau]])
    Y = np.array([nodes[y, 0:N]])
    Z_set = nodes[z_set]
    Z_set = Z_set[:, 0:N]
    # 样本空间K，包括节点x，y，Z的观测样本
    K_set = np.concatenate((X, Y, Z_set))
    # distance_1数组用于保存K中每一个样本相对其他样本的max-norm距离值。
    distance_1 = max_norm(K_set)
    # 备份distance_1的值，因为find_k_max函数调用后，原数组元素位置会发生改变。
    distance_2 = distance_1.copy()
    # 创建k_nearly数组保存第k邻近的样本在K_set中的index号。
    k_nearly = np.zeros(shape=N, dtype=np.int32)
    # 找到对于每个样本的第k邻近的样本，distance中有0，所以k+1。
    for i in range(0, N):
        k_min_value = find_k_min(distance_1[i], 0, len(distance_1[i])-1, k+1)
        k_nearly[i] = find_x(distance_2[i], k_min_value)
    # 创建epsilon数组，用于保存最后算出的epsilon值。
    epsilon = np.zeros(shape=N, dtype=np.float64)
    # 计算epsilon值，其值为目标样本与其第k临近的max-norm距离。
    for i in range(0, N):
        epsilon[i] = np.max(abs(K_set[:, i] - K_set[:, k_nearly[i]]))
    # 统计N_xz，N_yz和N_z
    N_xz = np.zeros(shape=N, dtype=np.int32)
    N_yz = np.zeros(shape=N, dtype=np.int32)
    N_z = np.zeros(shape=N, dtype=np.int32)
    # 用来储存xz，yz，z角度上的，样本间的max-norm距离
    distance_xz = max_norm(np.concatenate((X, Z_set)))
    distance_yz = max_norm(np.concatenate((Y, Z_set)))
    distance_z = max_norm(Z_set)
    # 找到各个角度上小于epsilon值的样本个数
    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                continue
            if distance_xz[i][j] < epsilon[i]:
                N_xz[i] += 1
            if distance_yz[i][j] < epsilon[i]:
                N_yz[i] += 1
            if distance_z[i][j] < epsilon[i]:
                N_z[i] += 1
    # 计算conditional_mutual_information,简称cm_info
    sum_of_phi = 0.0
    for i in range(0, N):
        sum_of_phi = sum_of_phi + phi_x[(N_xz[i] + 1)] + phi_x[(N_yz[i] + 1)] - phi_x[(N_z[i] + 1)]
    # k为取第k近邻
    cm_info = phi_x[(k)] - sum_of_phi / N
    return cm_info
