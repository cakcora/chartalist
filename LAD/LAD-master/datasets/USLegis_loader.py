import dateutil.parser as dparser
import re
import networkx as nx


'''
treat each year as a timestamp 
'''
def load_legis_temporarl_edgelist(fname):
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
	G = nx.Graph()

	for i in range(0, len(lines)):
		line = lines[i]
		values = line.split(',')
		t = values[0]
		v = int(values[1])		
		u = int(values[2])
		w = int(values[3]) 	#edge weight by number of shared publications in a year
		if current_date != '':
			if t != current_date:
				G_times.append(G)	#append old graph
				G = nx.Graph()	#create new graph
				current_date = t
		else:
			current_date = t
		G.add_edge(u, v, weight=w)
	G_times.append(G)
	print ("maximum time stamp is " + str(len(G_times)))
	return G_times

