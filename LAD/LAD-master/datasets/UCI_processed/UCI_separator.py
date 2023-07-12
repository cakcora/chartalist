import numpy as np
import networkx as nx
import pylab as plt
import dateutil.parser as dparser
import re

'''
treat each day as a discrete time stamp
'''
def load_temporarl_edgelist(fname, max_nodes=-1):
	edgelist = open(fname, "r")
	lines = list(edgelist.readlines())
	edgelist.close()
	#assume it is a directed graph at each timestamp
	# G = nx.DiGraph()

	#date u  v  w
	#find how many timestamps there are
	max_time = 0
	current_date = ''
	#create one graph for each day
	G_times = []
	G = nx.DiGraph()
	if(max_nodes > 0):
		G.add_nodes_from(list(range(0, max_nodes)))

	for i in range(0, len(lines)):
		line = lines[i]
		values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
		if (len(values) < 3):
			continue
		else:
			match = re.search(r'\d{4}-\d{2}-\d{2}', line)
			date_str = match.group(0)  #xxxx-xx-xx
			
			#start a new graph with a new date
			if (date_str != current_date):
				if (current_date != ''):
					G_times.append(G)	#append old graph
					G = nx.DiGraph()	#create new graph
					if(max_nodes > 0):
						G.add_nodes_from(list(range(0, max_nodes)))
				current_date = date_str		#update the current date

			w = int(values[-1]) 	#edge weight by number of characters 
			v = int(values[-2])		
			u = int(values[-3])
			G.add_edge(u, v, weight=w) 
	G_times.append(G)
	print ("maximum time stamp is " + str(len(G_times)))
	return G_times


def separate_files(G_times):
	for i in range(len(G_times)):
		G = G_times[i]
		t = i
		fname = "UCI/" + str(t) + ".txt"
		edgelist = open(fname, "w")
		for (u,v) in G.edges:
			edgelist.write(str(u) + " " + str(v) + " " + str(G[u][v]['weight']) + "\n")
		edgelist.close()
		print ("finished writing " + str(t))



	






def main():
	G_times = load_temporarl_edgelist("OCnodeslinks_chars.txt", max_nodes=1901)
	separate_files(G_times)






if __name__ == "__main__":
    main()
