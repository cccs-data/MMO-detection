import numpy as np

'''
    Define complementary method.
    input k_set and v_set
    return c_set = v_set - k_set
'''

def complementary_set(k_set, k_size, v_set):
    c_set = np.zeros(shape=len(v_set) - k_size, dtype=np.int64)
    c_size = 0
    for i in v_set:
        # If i in k_set, then skip it. Otherwsie, push it into c_set.
        if has_i(i, k_set, k_size):
            continue
        # push
        c_set[c_size] = i
        c_size += 1
    return c_set, c_size

'''
    has_i: find whether i is in k_set
'''

def has_i(i, k_set, k_size):
    for j in range(0, k_size):
        if k_set[j] == i:
            return True
    return False
