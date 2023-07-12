import numpy as np
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds, eigsh
from scipy import sparse
from numpy import linalg as LA
from datasets import UCI_loader
from datasets import SBM_loader
from datasets import USLegis_loader
from datasets import canVote_loader
from util import normal_util

def random_SVD(G_times, directed=True, num_eigen=6, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0


    for G in G_times:
        if (directed):
            A = nx.to_numpy_matrix(G)

        else:
            G2 = G.to_undirected()
            A = nx.to_scipy_sparse_matrix(G)
            A = A.asfptype()

        if (top):
            which="LM"
        else:
            which="SM"


        #compute svd, find diagonal matrix and append the diagonal entries
        #only consider 6 eigenvalues for now as the number of graph is small
        #num_eigenvalues=6
        #k=min(L.shape)-1
        u, s, vh = randomized_svd(A, num_eigen)
        vals = s
        vecs = u
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        #Temporal_eigenvalues.append(np.asarray(vals))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)


def find_eigs(G_times, max_size, directed=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0


    for G in G_times:
        A = nx.to_numpy_matrix(G)
        vals, vecs = LA.eig(A)
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))
        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)









'''
compute the eigenvalues for square laplacian matrix per time slice 
input: list of networkx Graphs
output: list of 1d numpy array of diagonal entries computed from SVD
'''
def SVD_perSlice(G_times, directed=True, num_eigen=6, top=True, max_size=500):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0

    for G in G_times:
        if (len(G) < max_size):
            for i in range(len(G), max_size):
                G.add_node(-1 * i)      #add empty node with no connectivity (zero padding)
        if (directed):
            L = nx.directed_laplacian_matrix(G)

        else:
            L = nx.laplacian_matrix(G)
            L = L.asfptype()

        if (top):
            which="LM"
        else:
            which="SM"

        u, s, vh = svds(L,k=num_eigen, which=which)
        # u, s, vh = randomized_svd(L, num_eigen)
        vals = s
        vecs = u
        #vals, vecs= LA.eig(L)
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)


'''
compute singular value decomposition for laplacian matrix per time slice
input: list of edgelists for each timestamp
reconstruct a large sparse matrix

limit to up to k eigenvalues
'''
def limited_eigenVal(G_times, directed=True, num_eigen=6, max_nodes=88282, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0

    #D are degree for each node at diagonal entries 
    #L = -A + D

    for G in G_times:

        #1. construct a 0 matrix
        L = np.zeros((max_nodes, max_nodes), dtype=np.int8)

        #2. add the adjacency edges (out edges)
        # -A
        for (u,v) in G:
            L[u,v] = L[u,v]-1

        #consider outdegrees
        #3. compute the out degree for each node
        # +D
        for i in range(0, max_nodes):
            L[i,i] = L[i,i] + np.sum(L[i])

        L = sparse.csr_matrix(L)
        L = L.asfptype()

        if (top):
            which="LM"
        else:
            which="SM"


        #compute svd, find diagonal matrix and append the diagonal entries
        u, s, vh = svds(L,k=num_eigen, which=which)
        vals = s
        vecs = u

        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)





'''
compute the SVD for adjacency matrix per time slice 
input: list of networkx Graphs
output: list of 1d numpy array of diagonal entries computed from SVD
'''
def adj_eigenvecs_perSlice(G_times, directed=True, num_eigen=6, top=True):
    Temporal_eigenvalues = []
    activity_vecs = []  #eigenvector of the largest eigenvalue
    counter = 0


    for G in G_times:
        if (directed):
            A = nx.to_numpy_matrix(G)

        else:
            G2 = G.to_undirected()
            A = nx.to_scipy_sparse_matrix(G)
            A = A.asfptype()

        if (top):
            which="LM"
        else:
            which="SM"
        u, s, vh = svds(A, k=num_eigen, which=which)
        vals = s
        vecs = u
        
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues.append(np.asarray(vals))

        print ("processing " + str(counter), end="\r")
        counter = counter + 1

    return (Temporal_eigenvalues, activity_vecs)



'''
Compute the SVD diagonal vectors and save them
'''
def compute_diags(outEigenFile, outVecFile, fname="datasets/OCnodeslinks_chars.txt", max_nodes=1901, UCI=True):
    if UCI:
        G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    else:
        G_times = DBLP_loader.load_dblp_temporarl_edgelist(fname, max_nodes=max_nodes)

    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=UCI)
    normal_util.save_object(Temporal_eigenvalues, outEigenFile)
    normal_util.save_object(activity_vecs, outVecFile)




def compute_adj_SVD(outEigenFile, outVecFile, fname="datasets/OCnodeslinks_chars.txt", max_nodes=1901, UCI=True):
    if UCI:
        G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    else:
        G_times = DBLP_loader.load_dblp_temporarl_edgelist(fname, max_nodes=max_nodes)
    (Temporal_eigenvalues, activity_vecs) = adj_eigenvecs_perSlice(G_times, directed=UCI)
    normal_util.save_object(Temporal_eigenvalues, outEigenFile)
    normal_util.save_object(activity_vecs, outVecFile)


def visiualize_vecs_UCI(eigen_file, vec_file, eigen_name, vec_name):
    Temporal_eigenvalues = normal_util.load_object(eigen_file)
    activity_vecs = normal_util.load_object(vec_file)
    limit = 5

    for i in range(0, len(Temporal_eigenvalues)):
        Temporal_eigenvalues[i] = Temporal_eigenvalues[i][0:limit]

    for i in range(0,len(activity_vecs)):
        activity_vecs[i] = activity_vecs[i].flatten()[0:limit]

    graph_name = "UCI"
    normal_util.plot_activity_intensity(np.asarray(Temporal_eigenvalues).real, eigen_name)
    normal_util.plot_activity_intensity(np.asarray(activity_vecs).real, vec_name)




def compute_synthetic_SVD(fname, num_eigen=499, top=True):
    
    edgefile = "datasets/SBM_processed/" + fname + ".txt"

    '''
    careful
    '''


    max_nodes = 1000




    max_time = 151
    directed = False

    G_times = SBM_loader.load_temporarl_edgelist(edgefile)

    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=directed, num_eigen=num_eigen, top=top, max_size=max_nodes)
    normal_util.save_object(Temporal_eigenvalues, fname+ ".pkl")



def compute_legis_SVD(num_eigen=6, top=True):
    fname = "datasets/USLegis_processed/LegisEdgelist.txt"
    directed = False

    G_times = USLegis_loader.load_legis_temporarl_edgelist(fname)
    max_nodes = 102
    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=directed, num_eigen=num_eigen, top=top, max_size=max_nodes)
    normal_util.save_object(Temporal_eigenvalues, "USLegis_L_singular.pkl")




def compute_canVote_SVD(num_eigen=338, top=True):
    fname = "datasets/canVote_processed/canVote_edgelist.txt"
    directed = True
    G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)
    max_len = 0
    for G in G_times:
        if (len(G) > max_len):
            max_len = len(G)
    print (max_len)
    max_nodes = max_len
    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=directed, num_eigen=num_eigen, top=top, max_size=max_nodes) 
    normal_util.save_object(Temporal_eigenvalues, "canVote_L_singular.pkl")
    

def compute_UCI_SVD(num_eigen=6, top=True):
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    directed = True
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    (Temporal_eigenvalues, activity_vecs) = SVD_perSlice(G_times, directed=directed, num_eigen=num_eigen, top=top, max_size=max_nodes)
    normal_util.save_object(Temporal_eigenvalues, "UCI_L_singular.pkl")

    





def main():
    compute_legis_SVD()

if __name__ == "__main__":
    main()
