import numpy as np
from KSG.ComplementarySet import complementary_set
from KSG.CMI import conditional_mutual_information

'''
    ADC algorithm:
        let z_set = empty and causal_entropy_p = inf, p = x.
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
    # x is the number of the target node, n is the number of samples, nodes is the array of all the data and t is the current time stamp (t>N).
    # create z_set to save the causal parents of node i, where z_set is a stack and r denotes the number of nodes.
    r = nodes.shape[0]
    z_set = np.zeros(shape=r, dtype=np.int64)
    z_size = 0
    # initiate p as node of x
    p = x
    # set causal entropy of p as inf
    causal_entropy_p = np.float64('inf')

    # judge wheteher p is a causal parent of x and find next causal parent
    while causal_entropy_p > 0 and z_size <= r:
        # if causal entropy > 0, p is a causal parent of i and push it into z_set
        z_set[z_size] = p
        z_size += 1

        # create c_set = {V - z_set}, c is a stack
        c_set, c_size = complementary_set(z_set, z_size, np.arange(0, r, 1))
        if c_size == 0:
            break
        # print('c_set = ', c_set)

        # create causal_entropy_j to save Cj-i|k
        causal_entropy_j = np.zeros(shape=r, dtype=np.float64)
        # calculate causal entropy
        for j in c_set:
            if j == x:
                continue
            # causal_entropy_j = cmi
            causal_entropy_j[j] = conditional_mutual_information(x, j, z_set[: z_size], nodes, k, tau)
            # print('x = ', x, 'j = ', j, causal_entropy_j[j])
        # next causal parent is the node with maximum Cj-i|k
        p = np.argmax(causal_entropy_j)
        causal_entropy_p = np.max(causal_entropy_j)
        # print(x, p, causal_entropy_p)

    # the past information of the target node is saved in Z.
    return z_set, z_size

