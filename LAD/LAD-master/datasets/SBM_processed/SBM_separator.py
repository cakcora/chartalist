'''
dataset loader for synthetic SBM dataset
undirected graphs
'''



import numpy as np
import networkx as nx
import pylab as plt
import dateutil.parser as dparser
import argparse
import re
import os

'''
treat each day as a discrete time stamp
'''
def load_temporarl_edgelist(fname, max_nodes=1000, max_time=150):
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
			G_times.append(G)	#append old graph
			G = nx.Graph()	#create new graph
			cur_t = t 
		G.add_edge(u, v) 
	G_times.append(G)
	print ("maximum time stamp is " + str(len(G_times)))
	return G_times



def separate_files(G_times, title):
	if not os.path.exists(os.path.join(os.getcwd(), title)):
		os.makedirs(os.path.join(os.getcwd(), title))

	for i in range(len(G_times)):
		G = G_times[i]
		t = i
		fname = title + "/" + str(t) + ".txt"
		edgelist = open(fname, "w")
		for (u,v) in G.edges:
			edgelist.write(str(u) + " " + str(v) + " " + str(1) + "\n")
		edgelist.close()
		print ("finished writing " + str(t), end="\r")



	






def main():
    parser = argparse.ArgumentParser(description='parse egdelist for EdgeMonitoring')
    parser.add_argument('-f','--file', 
                    help='which file to parse', required=True)
    args = vars(parser.parse_args())
    title = args["file"]
    G_times = load_temporarl_edgelist(title + ".txt", max_nodes=500, max_time=151)
    separate_files(G_times, title)


if __name__ == "__main__":
    main()



