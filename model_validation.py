import igraph
import KOrderPathModel
import pathpy as pp
from math import ceil

#define parameters of the generated dataset
#n: number of the nodes, deg: degree of each node
n = 20
deg = 5

detected_order = [[1] for _ in range(3)]


for order in range(2, 5):
    # generate random (strongly connected) network
    g = igraph.Graph.Erdos_Renyi(n=n, m=int(n*deg), directed=True, loops=True).clusters(mode='STRONG').giant()
    # generate k-th-order path model
    pathModel = KOrderPathModel.KOrderPathModel(g, k=int(order))
    batch = 1
    while batch <= 20:
        try:
            pathset = pathModel.generatePaths(pathCount=ceil(10**(0.25*batch)), pathLength=20)
            
            model = pp.MultiOrderModel(pathset, max_order=order+1)
            #estimate optimal order
            optimal_order = model.estimate_order()
            print('k = ' + str(order) + ', batch = ' + str(batch) + ':')
            print('the optimal Markovian order of the data is '+str(optimal_order))
            detected_order[order - 2].append(optimal_order)
        except:
            print('k = ' + str(order) + ', batch = ' + str(batch) + ' PathsTooShort')
            print('retrying...')
        else:
            batch += 1

print(detected_order)