import pathpy
import numpy as np
from pickle import load
from utils import break_and_add_path

#load temporal networks, whose format is (timestamps * agents * agents)
f = open('./result/oCEP_output','rb')
raw_TN = load(f)
f.close()

#normalize the temporal networks
raw_TN = raw_TN / np.amax(raw_TN)

#define the number of the agent (individuals) in the network
#agents = 10
agents = raw_TN.shape[1]

#construct temporalnetwork TN based on raw data
#the individuals are identified as p0, p1, ...
#The threshold is set as 0.1, any path with weitht >= 0.1 is regarded as connected
TN = pathpy.TemporalNetwork()
for i in range(raw_TN.shape[0]):
    for j in range(agents):
        for k in range(agents):
            if raw_TN[i][j][k] >= 0.1:
                TN.add_edge('p'+str(j), 'p'+str(k), i)

#show information of the temporal network
print('basic information of the temporal networks:')
print(TN)

#extract raw pathset from temporal networks
print('\nextracting pathset from temporal network...\n')
S = pathpy.path_extraction.paths_from_temporal_network_dag(TN, delta = 1)

#show basic information of the pathset
print('information of the original pathset:')
print(S)

#convert pathset as lists
print('extracting path information')
original_path_set = S.sequence()
original_path_set = '+'.join(original_path_set)
original_path_set = original_path_set.split('|')
original_path_set.pop()
for i in range(len(original_path_set)):
    temp = original_path_set[i].split('+')
    temp.remove('')
    if i != 0:
        temp.remove('')
    original_path_set[i] = temp

#extract paths without redundant nodes
real_path_set = pathpy.Paths()
print('generating pathset without redundant nodes...')
break_and_add_path(original_path_set, real_path_set)

print('information of the pathset without redundant nodes:')
print(real_path_set)

#The high-order model is generated from the path set S and named as Model. The maximum order is preliminarily set as 5. If the final estimated optimal order is the same as the maximum order, the maximum order should be increased
max_order = 5
success = False
while not success:
    try:
        model = pathpy.MultiOrderModel(real_path_set, max_order=max_order)
        success = True
    except:
        max_order -= 1

#estimate optimal order of the pathset
print('detecting the optimal order...')
optimal_order = model.estimate_order()
print('the optimal Markovian order of the data is ' + str(optimal_order))
