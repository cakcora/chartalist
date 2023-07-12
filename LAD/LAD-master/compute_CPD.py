import numpy as np
import tensorly as tl
import networkx as nx
from tensorly.decomposition import parafac
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from datasets import UCI_loader
from datasets import SBM_loader
from datasets import USLegis_loader
from datasets import canVote_loader

from util import normal_util
import re
import pickle
import datetime

def compute_accuracy(anomalies, real_events):

    correct = 0
    for anomaly in anomalies:
        if anomaly in real_events:
            correct = correct + 1

    return (correct/len(real_events))


'''
G_times: is a temporal graph where each element in the list is a networkx graph
return a third order tensor T
'''
def toTensor(G_times):
    T = []
    #load adjacency matrix from each time step and add it to tensor
    for G in G_times:
        A = nx.to_numpy_matrix(G)
        A = np.resize(A, (100,100))
        A = np.asarray(A)
        A.astype(float)
        T.append(A)

    T = tl.tensor(T)
    return T 

'''
apply parafac decomposition on tensor
'''
def apply_parafac(T, dimension=3):
    factors = parafac(T, rank=dimension)
    print ("there are " + str(len(factors)))
    # print (factors[1])
    print ([f.shape for f in factors[1]])
    return factors



def find_factors_UCI():
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    T = toTensor(G_times, max_nodes)
    dim = 3
    print ("CPD starts")
    print (datetime.datetime.now())
    factors = apply_parafac(T, dimension=dim)
    print (datetime.datetime.now())
    print ("CPD ends")
    tname = "UCI_factors.pkl"
    normal_util.save_object(factors,tname)


def LocalOutlierFactor_anomalies(factors, n_neighbors=20):
    anomalies = []
    Temporal_factors = factors[1][0]
    print (Temporal_factors.shape)
    
    total_t = len(Temporal_factors)
    '''
    Use the LocalOutlierFactor algorithm from Sklearn 
    uses k nearest neighbor to detect outliers
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
    '''
    #n_neighbors = total_t  #start with 20  
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    predictions = clf.fit_predict(Temporal_factors)
    for i in range(total_t):
        if (predictions[i] == -1):
            anomalies.append(i)
    return anomalies


    '''
    eps are the maximum distance between two samples to be considered as neighbors
    min_samples are the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    '''
def DBSCAN_anomalies(factors, eps=3, min_samples=2, min_size=10):
    anomalies = []
    Temporal_factors = factors[1][0]
    print (Temporal_factors.shape)
    total_t = len(Temporal_factors)


    clf = DBSCAN(eps=eps, min_samples=min_samples)
    predictions = clf.fit_predict(Temporal_factors)

    '''
    Use the DBSCAN algorithm from Sklearn 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit_predict
    '''
    count_labels = {}   
    for label in predictions:
        if label not in count_labels:
            count_labels[label] = 1
        else:
            count_labels[label] = count_labels[label] + 1

    rare_labels = []        #find all clusters that have size smaller than 5
    for key in count_labels:
        if (count_labels[key] <= min_size):
            rare_labels.append(key)

    rare_labels.append(-1)

    #-1 is when it doesn't fit any density centers
    for i in range(total_t):
        if (predictions[i] in rare_labels):
            anomalies.append(i)
    return anomalies




def find_synthetic_factors(fname):
    fname = "datasets/SBM_processed/" + fname + ".txt" 
    max_nodes = 500
    num_timestamps = 151
    G_times = SBM_loader.load_temporarl_edgelist(fname)
    T = toTensor(G_times)
    dim = 30
    print ("CPD starts")
    print (datetime.datetime.now())
    factors = apply_parafac(T, dimension=dim)
    normal_util.save_object(factors, "SBM_factors" + str(dim) +".pkl")
    print (datetime.datetime.now())
    print ("CPD ends")

    #factors = normal_util.load_object("SBM_factors30.pkl")

    real_events = [16,31,61,76,91,106,136]

    '''
    either can be an option here
    '''
    anomalies = LocalOutlierFactor_anomalies(factors, n_neighbors=20)
    #anomalies = DBSCAN_anomalies(factors, eps=3, min_samples=2, min_size=10)
    accuracy = compute_accuracy(anomalies, real_events)
    print (anomalies)
    print ("prediction accuracy is " + str(accuracy))


def find_UCI_factors():
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    num_timestamps = 196
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    T = toTensor(G_times)
    dim = 30
    print ("CPD starts")
    print (datetime.datetime.now())
    factors = apply_parafac(T, dimension=dim)
    normal_util.save_object(factors, "UCI_factors" + str(dim) +".pkl")
    print (datetime.datetime.now())
    print ("CPD ends")
    #factors = normal_util.load_object("UCI_factors1000.pkl")
    
    real_events = [65,158]
    anomalies = DBSCAN_anomalies(factors, eps=3, min_samples=2, min_size=10)
    #anomalies = LocalOutlierFactor_anomalies(factors, n_neighbors=20)
    accuracy = compute_accuracy(anomalies, real_events)
    print (anomalies)
    print ("prediction accuracy is " + str(accuracy))


def find_USLegis_factors():
    fname = "datasets/USLegis_processed/LegisEdgelist.txt"
    G_times = USLegis_loader.load_legis_temporarl_edgelist(fname)
    T = toTensor(G_times)
   
    dim = 10
    print ("CPD starts")
    print (datetime.datetime.now())
    factors = apply_parafac(T, dimension=dim)
    print (datetime.datetime.now())
    print ("CPD ends")
    normal_util.save_object(factors, "USLegis_factors" + str(dim) +".pkl")
    
    real_events = [3,7]

    anomalies = LocalOutlierFactor_anomalies(factors, n_neighbors=5)
    accuracy = compute_accuracy(anomalies, real_events)
    print (anomalies)
    print ("prediction accuracy is " + str(accuracy))


def find_canVote_factors():
    fname = "datasets/canVote_processed/canVote_edgelist.txt"
    G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)

    T = toTensor(G_times)
    dim = 10
    print ("CPD starts")
    print (datetime.datetime.now())
    factors = apply_parafac(T, dimension=dim)
    print (datetime.datetime.now())
    print ("CPD ends")
    normal_util.save_object(factors, "canVote_factors" + str(dim) +".pkl")
    
    anomalies = LocalOutlierFactor_anomalies(factors, n_neighbors=7)
    print (anomalies)



def main():

    find_synthetic_factors("eventCP_0.05_0.25_1.0")
    #find_UCI_factors()
    #find_canVote_factors()
    #find_USLegis_factors()
    


if __name__ == "__main__":
    main()
