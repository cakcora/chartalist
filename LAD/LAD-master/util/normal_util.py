import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import networkx as nx
from scipy import sparse
import pylab as plt
import dateutil.parser as dparser
import re


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 2)

def load_object(filename):
	output = 0
	with open(filename, 'rb') as fp:
		output = pickle.load(fp)
	return output

'''
convert edgelist to number of edges
'''
def edgelist2numEdge(edgelist):
	num_edges = len(set(edgelist))
	return num_edges

'''
convert edgelist to weight list
'''
def edgelist2weights(edgelist):
	unique_edges = list(set(edgelist))
	weights = [0]*len(unique_edges)
	for edge in edgelist:
		weights[unique_edges.index(edge)] += 1
	return weights

def edgelist2degrees(edgelist):
	unique_nodes = []
	for (u,v) in edgelist:
		if u not in unique_nodes:
			unique_nodes.append(u)
		if v not in unique_nodes:
			unique_nodes.append(v)
	
	degrees = [0]*len(unique_nodes)
	for (u,v) in edgelist:
		degrees[unique_nodes.index(u)] +=1
		degrees[unique_nodes.index(v)] +=1
	return degrees


def plot_ranks(anoRank, proRank, fname):
	t = list(range(0, len(anoRank)))
	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	colors = ['#ffa600', '#003f5c', '#bc5090', '#e5f5e0','#fa9fb5','#c51b8a']
	ax.plot(t, anoRank, marker="P", color=colors[0], ls='solid', linewidth=0.5, markersize=1, label="Anomaly Rank")
	ax.plot(t, proRank, marker="P", color=colors[1], ls='solid', linewidth=0.5, markersize=1, label="Graph property Rank")



	ax.set_xlabel('time stamps', fontsize=8)
	# ax.set_yscale('log')
	ax.set_ylabel('rank', fontsize=8)
	plt.title("plotting rank over time", fontsize='small')
	plt.legend(fontsize=3)
	plt.savefig(fname+'anomalyScores.pdf')

	print ("plotting rank complete")






'''
find obvious outlier points in a line
by global average
find points that are outside of coefficient * standard deviation away from mean
coefficient is how many standard deviation away should the outlier be considered
'''
def find_global_average_outlier(x, coefficient=2.0):
	x = np.asarray(x)
	avg = np.mean(x)
	std = np.std(x)
	outlier = []
	for i in range(len(x)):
		if (abs(x[i]-avg) >= coefficient*std):
			outlier.append(i)
	return outlier



'''
find obvious outlier points in a line
by moving window average
find points that are outside of 1.5 standard deviation away from mean
coefficient is how many standard deviation away should the outlier be considered
percent_ranked is the percentage of outliers ranked by their moving std 
window is the sliding window size
initial_period is the starting time points which we don't consider as anomalies
'''
def find_local_average_outlier(x, coefficient=2.0, percent_ranked=0.05, window=5, initial_period=10):
	x = np.asarray(x)
	outliers = []
	for i in range(initial_period,len(x)):
		#compute moving average until this point
		avg = np.mean(x[i-window:i])
		std = np.std(x[i-window:i])
		if (abs(x[i]-avg) >= coefficient*std):
			outliers.append(i)

	return outliers



'''
find obvious outlier points in a line
by rarity and moving window average
find points that are outside of 1.5 standard deviation away from mean
coefficient is how many standard deviation away should the outlier be considered
percent_ranked is the percentage of outliers ranked by their moving std 
window is the sliding window size
initial_period is the starting time points which we don't consider as anomalies
'''
def find_rarity_windowed_outlier(x, percent_ranked=0.05, window=5, initial_period=10):
	#percent_ranked = 0.18
	x = np.asarray(x)
	mv_std = []

	for i in range(0, initial_period):
		mv_std.append(0)

	for i in range(initial_period,len(x)):
		#compute moving average until this point
		avg = np.mean(x[i-window:i])
		std = np.std(x[i-window:i])
		if (std == 0):
			std = 1
		mv_std.append(abs(x[i]-avg) / std)
		
	mv_std = np.asarray(mv_std)
	num_ranked = int(round(len(x) * percent_ranked))
	outliers = mv_std.argsort()[-num_ranked:][::-1]

	return outliers





'''
plot the number of edge per time slice and cumulatively
return outlier
'''
def plot_edges(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	num_edges = []
	cumulative_edges = []
	sum_edges = 0

	for G in G_times:
		if (isinstance(G, list)):
			num_edge = edgelist2numEdge(G)
		else:
			num_edge = G.number_of_edges()
		num_edges.append(num_edge)
		sum_edges = sum_edges + num_edge
		cumulative_edges.append(sum_edges)

	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, cumulative_edges, marker='o', color='#74a9cf', ls='solid', linewidth=0.5, markersize=1, label="cumulative edges")
	ax.plot(t, num_edges, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1, label="number of edges")
	ax.set_xlabel('time stamp', fontsize=8)

	'''
	plot outlier as vertical lines
	'''
	#outliers = find_global_average_outlier(num_edges)
	#outliers = find_local_average_outlier(num_edges)
	outliers = find_rarity_windowed_outlier(num_edges)
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)


	# ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('number of edges', fontsize=8)
	plt.title("plotting number of edges", fontsize='x-small')
	plt.legend(fontsize = 'x-small')
	plt.savefig(fname+'edge.pdf', pad_inches=0)

	return outliers

'''
make a plot to compare the three different methods of obtaining weak labels
'''
def plot_compare_weak_labels_edge(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	num_edges = []
	cumulative_edges = []
	sum_edges = 0

	for G in G_times:
		num_edges.append(G.number_of_edges())
		sum_edges = sum_edges + G.number_of_edges()
		cumulative_edges.append(sum_edges)

	fig, axs = plt.subplots(3)
	plt.rcParams.update({'figure.autolayout': True})
	plot_num = 0
	for ax in axs:
		ax.plot(t, cumulative_edges, marker='o', color='#74a9cf', ls='solid', linewidth=0.5, markersize=1, label="cumulative edges")
		ax.plot(t, num_edges, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1, label="number of edges")
		ax.set_yscale('log')
		ax.set_ylabel('number of edges', fontsize=5)
		ax.tick_params(axis="x", labelsize=5)
		ax.tick_params(axis="y", labelsize=5)
		'''
		plot outlier as vertical lines
		'''
		if (plot_num == 0):
			outliers = find_global_average_outlier(num_edges)
			ax.set_title("Global Average Outliers", fontsize=6)

		if (plot_num == 1):
			outliers = find_local_average_outlier(num_edges)
			ax.set_title("Moving Window Average Outliers", fontsize=6)

		if (plot_num == 2):
			outliers = find_rarity_windowed_outlier(num_edges)
			ax.set_title("Moving Window Rarity Outliers", fontsize=6)
			ax.set_xlabel('time stamp', fontsize=5)

		for xc in outliers:
			ax.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
		plot_num = plot_num + 1

		ax.legend()
	plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	plt.tight_layout()
	plt.savefig(fname+'comparison.pdf')



'''
plot avg_clustering_coefficient over time
'''
def plot_avg_clustering(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	avg_clustering = []

	for G in G_times:
		avg_clustering.append(nx.average_clustering(G))

	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, avg_clustering, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1)
	ax.set_xlabel('time', fontsize=8)

	outliers = find_rarity_windowed_outlier(avg_clustering, percent_ranked=0.05, window=5, initial_period=10)
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)

	# ax.set_xscale('log')
	# ax.set_yscale('log')
	ax.set_ylabel('average clustering coefficient', fontsize=8)
	plt.title("plotting temporal average clustering coefficient ", fontsize='x-small')
	#plt.legend(fontsize=8)
	plt.savefig(fname+'clustering.pdf', pad_inches=0)

	return outliers

'''
plot number of connected components for undirected graph
'''
def plot_num_components_undirected(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	num_connected_components = []

	for G in G_times:
		G = G.to_undirected()
		num_connected_components.append(nx.number_connected_components(G))

	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, num_connected_components, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1)
	ax.set_xlabel('time', fontsize=8)

	outliers = find_rarity_windowed_outlier(num_connected_components)
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	ax.set_ylabel('number of connected components', fontsize=8)
	plt.title("number of connected components over time", fontsize='x-small')
	plt.savefig(fname+'components.pdf', pad_inches=0)

	return outliers


'''
plot number of strongly and weakly connected components for directed graph
'''
def plot_num_components_directed(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	num_strong = []
	num_weak = []

	for G in G_times:
		num_strong.append(nx.number_strongly_connected_components(G))
		num_weak.append(nx.number_weakly_connected_components(G))

	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, num_strong, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="strongly")
	ax.plot(t, num_weak, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="weakly")
	ax.set_xlabel('time', fontsize=8)


	outliers = find_rarity_windowed_outlier(num_weak)		#use weakly 
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	ax.set_ylabel('number of connected components', fontsize=8)
	plt.title("number of connected components over time", fontsize='x-small')
	plt.legend(fontsize=5)
	plt.savefig(fname+'components.pdf', pad_inches=0)

	return outliers


'''
plot the average weight of edges per slice 
and maximum weight of edges per slice
and minimum weight of edges per slice
'''
def plot_weighted_edges(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	avg_weight = []
	max_weight = []
	min_weight = []
	for G in G_times:
		if (isinstance(G, list)):
			weights = edgelist2weights(G)
		else:
			weights=list(nx.get_edge_attributes(G,'weight').values())
		
		max_weight.append(max(weights))
		min_weight.append(min(weights))
		avg_weight.append(sum(weights) / len(weights))
	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, max_weight, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="maximum weight")
	ax.plot(t, min_weight, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="minimum weight")
	ax.plot(t, avg_weight, marker="o", color='#bc5090', ls='solid', linewidth=0.5, markersize=1, label="average weight")

	outliers = find_rarity_windowed_outlier(avg_weight)		#use weakly 
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)

	ax.set_xlabel('time stamp', fontsize=8)
	# ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('edge weight', fontsize=8)
	plt.title("weighted edges over time", fontsize='small')
	#plt.legend(bbox_to_anchor=(1.04,1), loc="lower center", fontsize='x-small')
	plt.legend(fontsize=5)
	plt.savefig(fname+'weighted.pdf')

	return outliers


'''
plot the degree sequences of the temporal graph
average degree 
and std of the degree
and max degree
and min degree
'''
def plot_degree_changes(G_times, fname):
	max_time = len(G_times)
	t = list(range(0, max_time))
	avg_degree = []
	# std_degree = []
	max_degree = []
	min_degree = []
	for G in G_times:
		degrees = []
		if (isinstance(G, list)):
			degrees = edgelist2degrees(G)
		else:
			for n in G.nodes:
				degrees.append(G.degree(n))

		
		max_degree.append(max(degrees))
		min_degree.append(min(degrees))
		avg_degree.append(sum(degrees) / len(degrees))
	plt.rcParams.update({'figure.autolayout': True})
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')
	fig = plt.figure(figsize=(4, 2))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(t, max_degree, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="maximum degree")
	ax.plot(t, min_degree, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="minimum degree")
	ax.plot(t, avg_degree, marker="o", color='#bc5090', ls='solid', linewidth=0.5, markersize=1, label="average degree")
	#ax.errorbar(t, avg_degree, yerr=std_degree, marker="o", color='#bc5090', ls='solid', linewidth=0.5, markersize=1, label="average degree")

	outliers = find_rarity_windowed_outlier(avg_degree)		#use weakly 
	outliers.sort()
	for xc in outliers:
		plt.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)

	ax.set_xlabel('time stamp', fontsize=8)
	# ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('degree', fontsize=8)
	plt.title("plotting degree change over time", fontsize='small')
	plt.legend(fontsize=5)
	plt.savefig(fname+'degree.pdf')

	return outliers




'''
plot the activity vectors across time using color as intensity
'''
def plot_activity_intensity(diag_vecs, fname):
	# sub_vecs = diag_vecs[0:24]
	diag_vecs = np.transpose(np.asarray(diag_vecs))		#let time be x-axis
	diag_vecs = np.flip(diag_vecs, 0)
	ax = plt.gca()

	plt.xlabel('time stamp')
	plt.ylabel('rank')
	plt.imshow(diag_vecs, aspect='equal')
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# plt.colorbar(np.array(diag_vecs), cax=cax)
	
	plt.savefig(fname+'spectrogram.pdf')
	plt.clf()




def compare_spectrogram(Laplace_eigs, Laplace_vecs, adj_eigs, adj_vecs, fname):
	fig, axs = plt.subplots(4)
	plt.rcParams.update({'figure.autolayout': True})
	fig.set_facecolor("white")
	plot_num = 0
	for ax in axs:
		
		'''
		plot outlier as vertical lines
		'''
		if (plot_num == 0):
			diag_vecs = Laplace_eigs
			ax.set_title("Laplacian Spectrum (Proposed)")

		if (plot_num == 1):
			diag_vecs = Laplace_vecs
			ax.set_title("Laplacian principal eigenvector")

		if (plot_num == 2):
			diag_vecs = adj_eigs
			ax.set_title("Adjacency Spectrum")

		if (plot_num == 3):
			diag_vecs = adj_vecs
			ax.set_title("Adjacency principal eigenvector")
			ax.set_xlabel('time stamp')

		diag_vecs = np.transpose(np.asarray(diag_vecs))
		ax.set_ylabel('index')
		ax.imshow(diag_vecs, aspect='auto')
		plot_num = plot_num + 1
		# fig.colorbar(diag_vecs)

	plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	plt.tight_layout()
	plt.savefig(fname+'specCompare.pdf')
	plt.clf()
	


'''
plot all graph properties plots in one
designed for UCI message dataset

weighted_edges, clustering, num_components

'''
def all_plots_in_one(G_times, fname):

	max_time = len(G_times)
	t = list(range(0, max_time))
	avg_clustering = []

	fig, axs = plt.subplots(3)
	plt.rcParams.update({'figure.autolayout': True})
	plot_num = 0

	avg_weight = []
	max_weight = []
	min_weight = []
	avg_clustering = []
	num_strong = []
	num_weak = []

	for G in G_times:
		weights=list(nx.get_edge_attributes(G,'weight').values())
		max_weight.append(max(weights))
		min_weight.append(min(weights))
		avg_weight.append(sum(weights) / len(weights))
		avg_clustering.append(nx.average_clustering(G))
		num_strong.append(nx.number_strongly_connected_components(G))
		num_weak.append(nx.number_weakly_connected_components(G))

	for ax in axs:
		ax.tick_params(axis="x", labelsize=5)
		ax.tick_params(axis="y", labelsize=5)

		#plot weighted edge graph
		if (plot_num == 0):
			ax.plot(t, max_weight, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="maximum weight")
			ax.plot(t, min_weight, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="minimum weight")
			ax.plot(t, avg_weight, marker="o", color='#bc5090', ls='solid', linewidth=0.5, markersize=1, label="average weight")
			ax.set_yscale('log')
			ax.set_ylabel('edge weight', fontsize=5)

			outliers = find_rarity_windowed_outlier(avg_weight)		#use weakly 
			outliers.sort()
			for xc in outliers:
				ax.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
			ax.set_title("weighted edges over time", fontsize='small')
			ax.legend(fontsize=5)


		#plot average clustering coefficinet
		if (plot_num == 1):
			ax.plot(t, avg_clustering, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1)

			outliers = find_rarity_windowed_outlier(avg_clustering, percent_ranked=0.05, window=5, initial_period=10)
			outliers.sort()
			for xc in outliers:
				ax.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
			ax.set_ylabel('average clustering coefficient', fontsize=5)
			ax.set_title("average clustering coefficient over time", fontsize='x-small')
			

		if (plot_num == 2):
			ax.plot(t, num_strong, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="strongly")
			ax.plot(t, num_weak, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="weakly")
			outliers = find_rarity_windowed_outlier(num_weak)		#use weakly 
			outliers.sort()
			for xc in outliers:
				ax.axvline(x=xc,color='k', linestyle=":", linewidth=0.5)
			ax.set_ylabel('number of connected components', fontsize=5)
			ax.set_title("number of connected components over time", fontsize='x-small')
			ax.set_xlabel('time stamp', fontsize=8)
			ax.legend()

		plot_num = plot_num + 1
	#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	plt.tight_layout()
	plt.savefig(fname+'allinOne.pdf')



'''
plot all graph properties plots in one
designed for a given dataset

weighted_edges, clustering, num_components
labels set is a list of list of labelled anomaly timestamps

'''
def all_in_one_compare(G_times, fname, label_sets, directed, window, initial_window, percent_ranked):

	max_time = len(G_times)
	t = list(range(0, max_time))
	avg_clustering = []

	fig, axs = plt.subplots(4)
	plt.rcParams.update({'figure.autolayout': True})
	plot_num = 0

	avg_weight = []
	max_weight = []
	min_weight = []
	avg_clustering = []
	# avg_shortest_path = []
	num_edges = []
	num_nodes = []

	if (directed):
		num_strong = []
		num_weak = []
	else:
		num_connected_components = []

	for G in G_times:
		weights=list(nx.get_edge_attributes(G,'weight').values())
		max_weight.append(max(weights))
		min_weight.append(min(weights))
		avg_weight.append(sum(weights) / len(weights))
		avg_clustering.append(nx.average_clustering(G))
		num_edges.append(G.number_of_edges())
		num_nodes.append(len(G))
		# avg_shortest_path.append(nx.average_shortest_path_length(G))

		if (directed):
			num_strong.append(nx.number_strongly_connected_components(G))
			num_weak.append(nx.number_weakly_connected_components(G))
		else:
			num_connected_components.append(nx.number_connected_components(G))

	for ax in axs:
		ax.tick_params(axis="x", labelsize=5)
		ax.tick_params(axis="y", labelsize=5)

		#plot weighted edge graph
		if (plot_num == 0):
			ax.plot(t, max_weight, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="maximum weight")
			ax.plot(t, min_weight, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="minimum weight")
			ax.plot(t, avg_weight, marker="o", color='#bc5090', ls='solid', linewidth=0.5, markersize=1, label="average weight")
			ax.set_yscale('log')
			ax.set_ylabel('edge weight', fontsize=5)

			outliers = find_rarity_windowed_outlier(avg_weight, percent_ranked=percent_ranked, window=window, initial_period=initial_window)		#use weakly 
			outliers.sort()
			colors = ['r','b','g','c','m','y']
			c_idx = 0
			# for xc in outliers:
			# 	ax.axvline(x=xc,color='k', linestyle='--', linewidth=1.0)
				
			for label_set in label_sets:
				for xc in label_set:
					ax.axvline(x=xc, color=colors[c_idx], linestyle="--", linewidth=1.0)
				c_idx = c_idx + 1


			ax.set_title("weighted edges over time", fontsize='small')
			ax.legend(fontsize=5)


		#plot average clustering coefficinet
		if (plot_num == 1):
			# ax.plot(t, avg_clustering, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1)
			ax.plot(t, num_nodes, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1)

			outliers = find_rarity_windowed_outlier(num_nodes, percent_ranked=percent_ranked, window=window, initial_period=initial_window)
			outliers.sort()
			colors = ['r','b','g','c','m','y']
			c_idx = 0
			# for xc in outliers:
			# 	ax.axvline(x=xc,color='k', linestyle="--", linewidth=1.0)
			for label_set in label_sets:
				for xc in label_set:
					ax.axvline(x=xc, color=colors[c_idx], linestyle="--", linewidth=1.0)
				c_idx = c_idx + 1

			
			ax.set_ylabel('number of nodes', fontsize=5)
			ax.set_title("number of active nodes over time", fontsize='x-small')

		#plot average shortest path
		if (plot_num == 2):
			ax.plot(t, num_edges, marker='o', color='#78f542', ls='solid', linewidth=0.5, markersize=1)

			outliers = find_rarity_windowed_outlier(num_edges, percent_ranked=percent_ranked, window=window, initial_period=initial_window)
			outliers.sort()
			colors = ['r','b','g','c','m','y']
			c_idx = 0
			# for xc in outliers:
			# 	ax.axvline(x=xc,color='k', linestyle="--", linewidth=1.0)
			for label_set in label_sets:
				for xc in label_set:
					ax.axvline(x=xc, color=colors[c_idx], linestyle="--", linewidth=1.0)
				c_idx = c_idx + 1

			
			ax.set_ylabel('number of edges', fontsize=5)
			ax.set_title("number of edges", fontsize='x-small')
			

		if (plot_num == 3):
			if (directed):
				ax.plot(t, num_strong, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1, label="strongly")
				ax.plot(t, num_weak, marker="h", color='#003f5c', ls='solid', linewidth=0.5, markersize=1, label="weakly")
			else:
				ax.plot(t, num_connected_components, marker="P", color='#ffa600', ls='solid', linewidth=0.5, markersize=1)
			
			if (directed):
				outliers = find_rarity_windowed_outlier(num_weak, percent_ranked=percent_ranked, window=window, initial_period=initial_window)		#use weakly 
			else:
				outliers = find_rarity_windowed_outlier(num_connected_components, percent_ranked=percent_ranked, window=window, initial_period=initial_window)
			outliers.sort()
			colors = ['r','b','g','c','m','y']
			c_idx = 0

			# for xc in outliers:
			# 	ax.axvline(x=xc,color='k', linestyle="--", linewidth=0.5)

			for label_set in label_sets:
				for xc in label_set:
					ax.axvline(x=xc, color=colors[c_idx], linestyle="--", linewidth=1.0)
				print ("set " + str(c_idx) + " is " + colors[c_idx])
				c_idx = c_idx + 1
			ax.set_ylabel('number of connected components', fontsize=5)
			ax.set_title("number of connected components over time", fontsize='x-small')
			ax.set_xlabel('time stamp', fontsize=8)

			if (directed):
				ax.legend()

		plot_num = plot_num + 1

	'''
	specify the xticks here
	'''
	labels = list(range(2006,2020))
	for i in range(len(labels)):
		labels[i] = str(labels[i])
	plt.xticks(t, labels)
	#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	plt.tight_layout()
	plt.savefig(fname+'allinOne.pdf')





	









