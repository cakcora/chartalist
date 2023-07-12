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

# def plot_nodes_edges(G_times, fname):
# 	max_time = len(G_times)
# 	t = list(range(0, max_time))
# 	num_nodes = []
# 	num_edges = []
# 	for G in G_times:
# 		num_nodes.append(G.number_of_nodes())
# 		num_edges.append(G.number_of_edges())

# 	plt.rcParams.update({'figure.autolayout': True})
# 	plt.rc('xtick', labelsize='x-small')
# 	plt.rc('ytick', labelsize='x-small')
# 	fig = plt.figure(figsize=(4, 2))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.plot(t, num_nodes, marker='o', color='#74a9cf', ls='solid', linewidth=0.5, markersize=1, label="nodes")
# 	ax.plot(t, num_edges, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1, label="edges")
# 	ax.set_xlabel('time stamp', fontsize=8)
# 	ax.set_xscale('log')
# 	ax.set_ylabel('number of nodes / edges', fontsize=8)
# 	plt.title("plotting number of nodes and edges in " + fname, fontsize='x-small')
# 	plt.legend(fontsize = 'x-small')
# 	plt.savefig("number of nodes and edges"+'.pdf',bbox_inches='tight', pad_inches=0)



# def main():
# 	fname = "UCI_message.edgelist.txt"
# 	G_times = load_temporarl_edgelist(fname)
# 	max_time = len(G_times)
# 	plot_nodes_edges(G_times, fname)



# if __name__ == "__main__":
#     main()
