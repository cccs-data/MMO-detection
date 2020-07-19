import numpy as np
from KSG.CMI import conditional_mutual_information
from KSG.ComplementarySet import complementary_set

'''
    PRNN algorithm：
        for every j in Z:
            if Cj-x|Z-{j} = 0 then:
                Z = Z - {j}
            end if 
        end for 

        return Ni

    Loop through all the nide y in Z. Calculate Cj-x|Z-{j} for each node Y. If Cj-x|Z-{j} = 0, then delete node Y.
'''


def PRNNAlgorithm(x, z_set, z_size, nodes, k, tau):
    # x is the target node. Z is the possible causal parents obtained by ADC (including x)
    # create stack to record nodes whose causal entropy is 0. These nodes will be deleted.
    need_to_delete = np.zeros(shape=z_size, dtype=np.int64)
    stack_size = 0
    causal_entropy = np.zeros(shape=nodes.shape[0], dtype=np.float64)

    # loop through Z to calculate causal entropy to detect the nodes to be deleted
    for j in range(1, z_size):
        # elete j from set z to form c_set
        c_set, c_size = complementary_set([z_set[j]], 1, z_set[0: z_size])
        # calculate causal entropy Cj-x|Z-{j}
        causal_entropy_c_set = conditional_mutual_information(x, z_set[j], c_set, nodes, k, tau)
        # judge whether causal entropy Cj-i|K-{j} is greater than 0. If Cj-i|K-{j} <= 0, the record j to be deleted latter.
        # print(causal_entropy_c_set)
        if causal_entropy_c_set <= 0:
            # record node j
            need_to_delete[stack_size] = j
            stack_size += 1
        else:
            causal_entropy[z_set[j]] = causal_entropy_c_set

    # delete all recorded nodes in K
    z_set = np.delete(z_set, need_to_delete[0: stack_size], axis=0)
    z_size = z_size - stack_size

    # return K after deleting the nodes
    return z_set, z_size, causal_entropy

