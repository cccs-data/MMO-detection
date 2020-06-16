"""
This file is from the 
"""

import random
import collections as co
import numpy as np
import pathpy as pp

class KOrderPathModel:
    
    def __init__(self, graph, k):
        """
        This class implements a Markov chain model of order k to 
        randomly generate paths in a given graph topology. For any 
        k > 1, the  topology of k-th order transitions will deviate 
        from what is expected based on a (k-1)-th order model

        @param graph: the graph topology on which paths shall be generated

        @param k: order of the Markov chain to generate paths
        """

        self.k = k       

        # generate higher-order nodes (i.e. all paths of length k-1)
        self.nodes = [(v.index,) for v in graph.vs()]

        for i in range(0, k-1):
            # expand nodes by edges step by step
            nodes_extended = list()
            for v in self.nodes:
                for w in graph.successors(v[-1]):
                        nodes_extended.append(v + (w,))
            self.nodes = nodes_extended

        
        # we now have one higher-order node for each possible path of length k-1       
        self.n = len(self.nodes)
        # print(nodes)    

        self.edges = []

        # generate sample lists for paths of length k
        self.P = co.defaultdict( lambda: list() )         
        for v in self.nodes:
            for w in graph.successors(v[-1]):
                if k>1:
                    for x in range(np.random.randint(0,2)):
                        self.P[v].append(w)
                        self.edges.append((v, v[1:] + (w,)))
                else:
                    self.P[v].append(w)
                    self.edges.append((v, v[1:] + (w,)))
        self.edges = list(set(self.edges))
        

    def generatePaths(self, pathCount=1, pathLength=10):
        """
        This method generates a given number of paths, each having a given length

        @param pathCount: the number of paths to generate

        @param pathLength: the (maximum) length of each path. Note that paths can be shorter 
            if the Markov chain runs into a state that does not allow for a further transition.
            In this case, the current path will be concluded. 
        """
        # generate a Paths object that will contain the paths
        paths = pp.Paths()

        for i in range(pathCount):           
            path = random.choice(self.nodes)

            memory = path

            # add missing steps to path of length pathLength
            for l in range(pathLength-self.k+1):
                
                if len(self.P[memory])>0:
                    path += (random.choice(self.P[memory]),)

                    # assign new memory based on last k nodes on path
                    memory = path[-self.k:]
                else:
                    break

            # Add to Paths object
            paths.add_path(path)

        return paths