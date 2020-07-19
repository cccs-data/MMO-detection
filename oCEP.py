import numpy as np
import pickle as pk
# 引入ADC方法。
from KSG.ADC import ADCAlgorithm
# 引入PRNN方法。
from KSG.PRNN import PRNNAlgorithm
# 引入数据存储模块
from KSG.DataWriter import data_writer_b

'''
    自定义
'''
# 文件名
file_name = 'input_data'
# 定义样本量N，时间窗口，即计算一次因果关联需要的样本量
N = 20
# interval时间窗口的间隔
interval = 10
# tau为时滞
tau = 2
# k近邻
k = 4

'''
     读取数据。
'''
file = open('./data/' + file_name, 'rb')
data = pk.load(file)
file.close()

print('组数据装填完毕')
node_n = data.shape[0]
time_n = data.shape[1]
print('节点数为', node_n)
print('总采样量为', time_n)
print('指标维度为', data.shape[2])

# t为计算次数
t = int((time_n - N - tau)/interval) + 1
print('计算次数为', t)
# 记录时间循环数
l = 0
cmi_result = np.zeros(shape=[t, node_n, node_n], dtype=np.float64)

for time_start in range(0, time_n - N - tau, interval):
    # time_start为时段开始时刻，time_end为时段结束时刻。
    time_end = time_start + N
    print('k =', k, 'sample:', time_start, '-', time_end)
    # nodes指向带入计算的数据部分
    nodes = data[:, time_start: time_end + tau]
    for i in range(data.shape[0]):
        '''
            运行ADC算法。
        '''
        k_set, k_size = ADCAlgorithm(i, nodes, k, tau)

        '''
            运行PRNN算法。
        '''
        k_set, k_size, causal_entropy = PRNNAlgorithm(i, k_set, k_size, nodes, k, tau)

        # print('节点', pigeon_number_i, '的因果父母为', k_set[1:], k_size - 1)
        print('#', end='')
        # 结果存入cmi_result
        cmi_result[l, :, i] = causal_entropy
    print('\n', cmi_result[l])
    l += 1

    # 存起来
    data_writer_b('./result/' + 'oCEP_output', cmi_result)