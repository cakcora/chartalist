'''
dataset generator for synthetic SBM dataset
'''
import numpy as np
import pylab as plt
import random
import dateutil.parser as dparser
from networkx.utils import *
import re
import networkx as nx
from networkx import generators
import copy


'''
t u v w
'''
def to_edgelist(G_times, outfile):

    outfile = open(outfile,"w")
    tdx = 0
    for G in G_times:
        
        for (u,v) in G.edges:
            outfile.write(str(tdx) + "," + str(u) + "," + str(v) + "\n")
        tdx = tdx + 1
    outfile.close()
    print("write successful")


'''
generate ER graph snapshot 
https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.generators.random_graphs.gnp_random_graph.html#networkx.generators.random_graphs.gnp_random_graph
'''
def ER_snapshot(G_prev, alpha, p):

    '''
    for all pairs of nodes, keep its status from time t-1 with 1-alpha prob and resample with alpha prob
    '''
    G_t = G_prev.copy()
    G_new = generators.gnp_random_graph(500, p, directed=False)
    n = 500
    for i in range(0,n):
        for j in range(i+1,n):
            #remain the same if prob > alpha
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i,j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i,j)
                if (not G_new.has_edge(i,j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i,j)
    return G_t








def SBM_snapshot(G_prev, alpha, sizes, probs):

    G_t = G_prev.copy()
    nodelist = list(range(0,sum(sizes)))
    G_new = nx.stochastic_block_model(sizes, probs, nodelist=nodelist)
    n = len(G_t)
    if (alpha == 1.0):
        return G_new

    for i in range(0,n):
        for j in range(i+1,n):
            #randomly decide if remain the same or resample
            #remain the same if prob > alpha
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i,j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i,j)
                if (not G_new.has_edge(i,j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i,j)
    return G_t


'''
blocks is an array of sizes
inter is the inter community probability
intra is the intra community probability
'''
def construct_SBM_block(blocks, inter, intra):
    probs = []
    for i in range(len(blocks)):
        prob = [inter]*len(blocks)
        prob[i] = intra
        probs.append(prob)
    return probs



'''
generate just change points
corresponding to the pure setting in experiment section
inter_prob = p_{ex}
intra_prob = p_{in}
alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''
def generate_pureSetting(inter_prob, intra_prob, alpha):
    cps=[15,30,60,75,90,105,135]
    fname = "pure_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []


    #let there be 500 nodes
    sizes_1 = [250,250] #500 nodes total at all times
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)


    sizes_3 = [50]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)


    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    sizes = sizes_2
    probs = probs_2
    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):
            if ((list_idx+1) > len(list_sizes)-1):
                list_idx = 0
            else:
                list_idx = list_idx + 1
            sizes = list_sizes[list_idx]
            probs = list_probs[list_idx]
            G_t = SBM_snapshot(G_t, 1.0, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)


'''
generate both change points and events
correspond to the hybrid setting in paper
inter_prob = p_{ex}
intra_prob = p_{in}
increment specifies the incremental for p_{ex} during events
alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''

def generate_hybridSetting(inter_prob, intra_prob, alpha, increment):
    cps=[15,30,60,75,90,105,135]
    fname = "hybrid_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []

    sizes_1 = [250,250] #500 nodes total at all times
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

    sizes_3 = [50]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)

    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    isEvent = True 
    sizes = sizes_2
    probs = probs_2

    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):

            if (isEvent):

                copy_probs = copy.deepcopy(probs)
                for i in range(len(copy_probs)):
                    for j in range(len(copy_probs[0])):
                        if (copy_probs[i][j] < intra_prob):
                            copy_probs[i][j] = copy_probs[i][j] + increment

                G_t = SBM_snapshot(G_t, 1.0, sizes, np.asarray(copy_probs))
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = False
                #go back to normal afterwards
            else:
                if ((list_idx+1) > len(list_sizes)-1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, 1.0, sizes, probs)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = True

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)





'''
generate just change points
inter_prob = p_{ex}
intra_prob = p_{in}
alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''
def generate_Connectivity_CP(inter_prob, intra_prob, increment, alpha=0.49):
    cps=[15,30,60,75,90,105,135]
    fname = "CPConnect_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(increment) + "_" + str(alpha) + ".txt"


    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)



    sizes = sizes_2
    probs = probs_2
    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):
            for i in range(len(probs)):
                for j in range(len(probs[0])):
                    if (probs[i][j] < intra_prob):
                        probs[i][j] = probs[i][j] + increment
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)





'''
generate just change points
inter_prob = p_{ex}
intra_prob = p_{in}
alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''
def generate_ChangePoint(inter_prob, intra_prob, alpha):
    cps=[15,30,60,75,90,105,135]
    fname = "ChangePoint_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []


    #let there be 500 nodes
    sizes_1 = [250,250] #500 nodes total at all times
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)


    sizes_3 = [50]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)


    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    sizes = sizes_2
    probs = probs_2
    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):
            if ((list_idx+1) > len(list_sizes)-1):
                list_idx = 0
            else:
                list_idx = list_idx + 1
            sizes = list_sizes[list_idx]
            probs = list_probs[list_idx]
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)






'''
generate both change points and events
inter_prob = p_{ex}
intra_prob = p_{in}
increment specifies the incremental for p_{ex} during events
alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''

def generate_event_change(inter_prob, intra_prob, alpha, increment):
    cps=[15,30,60,75,90,105,135]
    fname = "eventCP_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []

    sizes_1 = [250,250]
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [125,125,125,125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

    sizes_3 = [50]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)

    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    isEvent = True 
    sizes = sizes_2
    probs = probs_2

    maxt = 150
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):

            if (isEvent):

                copy_probs = copy.deepcopy(probs)
                for i in range(len(copy_probs)):
                    for j in range(len(copy_probs[0])):
                        if (copy_probs[i][j] < intra_prob):
                            copy_probs[i][j] = copy_probs[i][j] + increment

                G_t = SBM_snapshot(G_t, alpha, sizes, np.asarray(copy_probs))
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = False
                #go back to normal afterwards
            else:
                if ((list_idx+1) > len(list_sizes)-1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, alpha, sizes, probs)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = True

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)





def main():
    
    # inter_prob = 0.005
    # intra_prob = 0.080
    # increment = 0.01
    # alpha = 0.49 #recommended 0.49 in EdgeMonitoring
    # generate_Connectivity_CP(inter_prob, intra_prob, increment, alpha=alpha)



    inter_prob = 0.05
    intra_prob = 0.25
    increment = 0.10
    alpha = 0.0
    generate_pureSetting(inter_prob, intra_prob, alpha)
    #alpha = 0.1
    #generate_hybridSetting(inter_prob, intra_prob, alpha, increment)

    # inter_prob = 0.005
    # intra_prob = 0.03
    # increment = 0.10
    # alpha = 1.0
    # #generate_ChangePoint(inter_prob, intra_prob, alpha)
    # generate_event_change(inter_prob, intra_prob, alpha, increment)




    


if __name__ == "__main__":
    main()
