'''
dataset loader for synthetic SBM dataset
undirected graphs
'''



import numpy as np
import networkx as nx
import pylab as plt
import dateutil.parser as dparser
import re

'''
treat each day as a discrete time stamp
'''
def load_temporarl_edgelist(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    cur_t = 0

    '''
    t u v
    '''
    G_times = []
    G = nx.Graph()

    for i in range(0, len(lines)):
        line = lines[i]
        values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
        t = int(values[0])
        u = int(values[1])
        v = int(values[2])
        #start a new graph with a new date
        if (t != cur_t):
            G_times.append(G)   #append old graph
            G = nx.Graph()  #create new graph
            cur_t = t 
        G.add_edge(u, v) 
    G_times.append(G)
    print ("maximum time stamp is " + str(len(G_times)))
    return G_times