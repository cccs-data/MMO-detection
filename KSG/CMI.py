import numpy as np
from scipy.special import gamma
from scipy.misc import derivative


# phi(x) = gamma'(x)/gamma(x)。
def phi(x):
    t1 = derivative(gamma, x, dx=1e-12, n=1)
    t2 = gamma(x)
    return t1/t2


phi_x = np.zeros(shape=101, dtype=np.float64)
for i in range(1, 101):
    phi_x[i] = phi(i)


def max_norm(array):
    '''
    calculate max-norm distance array among samples
    '''
    N = array.shape[1]
    # create distance array to save the diastance between samples in K
    distance = np.zeros(shape=[N, N], dtype=np.float64)
    for i in range(0, N):
        # calculate max-norm distance
        for j in range(i+1, N):
            if i == j:
                continue
            distance[i][j] = np.max(np.abs(array[:, i] - array[:, j]))
            distance[j][i] = distance[i][j]
    return distance


def find_k_min(array, left, right, k):
    '''
    find the k-th largest element in array
    '''
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


def find_x(array, x):
    '''
    find the index of x
    '''
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
'''


def conditional_mutual_information(x, y, z_set, nodes, k, tau):
    N = nodes.shape[1]-tau
    # restore X, Y and Z_set
    X = np.array([nodes[x, tau:N+tau]])
    Y = np.array([nodes[y, 0:N]])
    Z_set = nodes[z_set]
    Z_set = Z_set[:, 0:N]
    # sample space K, including observed samples of node x, y and z
    K_set = np.concatenate((X, Y, Z_set))
    # distance_1 is used to save max-norm distance of K
    distance_1 = max_norm(K_set)
    distance_2 = distance_1.copy()
    # create k_nearly to save the index of the k-th neareast neibour
    k_nearly = np.zeros(shape=N, dtype=np.int32)
    # find the k-th neareast neighbour of each sample
    for i in range(0, N):
        k_min_value = find_k_min(distance_1[i], 0, len(distance_1[i])-1, k+1)
        k_nearly[i] = find_x(distance_2[i], k_min_value)
    epsilon = np.zeros(shape=N, dtype=np.float64)
    # calculate epsilon
    for i in range(0, N):
        epsilon[i] = np.max(abs(K_set[:, i] - K_set[:, k_nearly[i]]))
    # count N_xz, N_yz and N_z
    N_xz = np.zeros(shape=N, dtype=np.int32)
    N_yz = np.zeros(shape=N, dtype=np.int32)
    N_z = np.zeros(shape=N, dtype=np.int32)
    # save corresponding max-norm distance between samples
    distance_xz = max_norm(np.concatenate((X, Z_set)))
    distance_yz = max_norm(np.concatenate((Y, Z_set)))
    distance_z = max_norm(Z_set)
    # find the number of the samples that is smaller than epsilon
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
    # calculate conditional_mutual_information (cm_info for short)
    sum_of_phi = 0.0
    for i in range(0, N):
        sum_of_phi = sum_of_phi + phi_x[(N_xz[i] + 1)] + phi_x[(N_yz[i] + 1)] - phi_x[(N_z[i] + 1)]
    # k is the k-th neareast neighbour
    cm_info = phi_x[(k)] - sum_of_phi / N
    return cm_info
