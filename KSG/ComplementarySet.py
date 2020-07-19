import numpy as np

'''
    定义complementary方法。
    输入一个k_set，全集v_set
    返回c_set = v_set - k_set。
'''

def complementary_set(k_set, k_size, v_set):
    # 创建c_set存储补集，c_set是一个栈。
    c_set = np.zeros(shape=len(v_set) - k_size, dtype=np.int64)
    c_size = 0
    # 定义全集为所有节点,遍历全集，找到补集。
    for i in v_set:
        # 如果k_set中有i节点，就跳过，没有就加入c_set。
        if has_i(i, k_set, k_size):
            continue
        # 入栈。
        c_set[c_size] = i
        c_size += 1
    return c_set, c_size

'''
    定义HasI方法。
    简而言之，就是找寻k_set中有没有i节点。
    有就返回True。
    没有就返回False。
'''

def has_i(i, k_set, k_size):
    for j in range(0, k_size):
        if k_set[j] == i:
            return True
    return False
