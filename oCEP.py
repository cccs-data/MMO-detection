import numpy as np
import pickle as pk
from KSG.ADC import ADCAlgorithm
from KSG.PRNN import PRNNAlgorithm
from KSG.DataWriter import data_writer_b

'''
    define parameters
'''
# file name
file_name = 'input_data'
# define the number of samples and the size of time window, which is the number of samples needed to calculate a causal matrix
N = 20
# interval is the interval of the time window
interval = 10
# tau is the time delay
tau = 2
# k in the KNN algorithm
k = 4

'''
     read data
'''
file = open('./data/' + file_name, 'rb')
data = pk.load(file)
file.close()

print('The data has been loaded.')
node_n = data.shape[0]
time_n = data.shape[1]
print('The number of individuals is', node_n)
print('The number of time stamps is', time_n)
print('The dimension of the data is', data.shape[2])

# t is the number of the causal matrix
t = int((time_n - N - tau)/interval) + 1
print('The number of generated causal matirx is', t)
# record number of loop
l = 0
cmi_result = np.zeros(shape=[t, node_n, node_n], dtype=np.float64)

for time_start in range(0, time_n - N - tau, interval):
    # time_start is the time to start prediction and time_end is the time to end prediction.
    time_end = time_start + N
    print('k =', k, 'sample:', time_start, '-', time_end)
    nodes = data[:, time_start: time_end + tau]
    for i in range(data.shape[0]):
        '''
            run ADC algorithm.
        '''
        k_set, k_size = ADCAlgorithm(i, nodes, k, tau)

        '''
            run PRNN algorithm.
        '''
        k_set, k_size, causal_entropy = PRNNAlgorithm(i, k_set, k_size, nodes, k, tau)

        print('#', end='')
        # save the result into cmi_result
        cmi_result[l, :, i] = causal_entropy
    print('\n', cmi_result[l])
    l += 1

    # save the result to file oCEP_output
    data_writer_b('./result/' + 'oCEP_output', cmi_result)