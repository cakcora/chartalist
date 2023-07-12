from util import normal_util
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout
import re



from datasets import UCI_loader
from datasets import SBM_loader
from datasets import USLegis_loader
from datasets import canVote_loader



def plot_UCI():

    '''
    plot statistics from UCI Message
    '''
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)

    graph_name = "UCI_Message"
    '''
    dictionary of weak labels
    '''
    labels_dict = {}
    print ("edge")
    labels_dict['edge'] = normal_util.plot_edges(G_times, graph_name)
    print ("acc")
    labels_dict['acc'] = normal_util.plot_avg_clustering(G_times, graph_name)
    print ("component")
    labels_dict['component'] = normal_util.plot_num_components_directed(G_times, graph_name)
    print ("weights")
    labels_dict['weights'] = normal_util.plot_weighted_edges(G_times, graph_name)
    print ("degree")
    labels_dict['degree'] = normal_util.plot_degree_changes(G_times, graph_name)
    return labels_dict

def plot_UCI_allinOne():
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)

    LAD = [69,70,184,185,187,188,189,191,192,194]
    activity = [57,75,78,85,89,90,104,176,188,192]
    CPD = [13,16,17,20,22,23,24,31,38,40,124]
    label_sets = []
    label_sets.append(LAD)
    label_sets.append(activity)
    label_sets.append(CPD)

    graph_name = "UCI_Message"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, True)







    graph_name = "UCI_Message"
    normal_util.all_plots_in_one(G_times, graph_name)

def print_labels(labels_dict):
    for label in labels_dict:
        print (label)
        print (labels_dict[label])






def plot_synthetic():
    fname = "datasets/SBM_processed/config_edgelist.txt"
    #fname = "datasets/SBM_processed/ER_synthetic_edgelist_sudden_0.002_0.3.txt"
    max_nodes = 100
    max_time = 150
    G_times = SBM_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes, max_time=max_time)
    graph_name = "synthetic"
    outliers = normal_util.plot_edges(G_times, graph_name)
    normal_util.plot_num_components_undirected(G_times,  graph_name)
    print (outliers)


def plot_legislative_allinOne():
    fname = "datasets/USLegis_processed/LegisEdgelist.txt"
    G_times = USLegis_loader.load_legis_temporarl_edgelist(fname)
    LAD = [3,7]
    label_sets = []
    label_sets.append(LAD)

    graph_name = "USLegislative"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, False)


def plot_canVote_allinOne():
    fname = "datasets/canVote_processed/canVote_edgelist.txt"
    G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)
    LAD = [2,7,11]
    label_sets = []
    label_sets.append(LAD)
    window = 1
    initial_window = 2
    percent_ranked = 0.2

    graph_name = "canVote"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, True, window, initial_window, percent_ranked)


def plot_spectrum(pkl_name, graph_name):
    eigen_slices = normal_util.load_object(pkl_name)
    normal_util.plot_activity_intensity(eigen_slices, graph_name)


def plot_vis(G):
    pos = nx.spring_layout(G)

    node_list = []
    for node in G:
        if (G.degree[node] > 50):
            node_list.append(node)
    print (len(node_list))

    print(node_list)

    colors = range(len(G))
    options = {
        "nodelist" : node_list,
        "node_size" : 5,
        "node_color": "#ffa600",
        "edge_color": "#ff6361",
        "width": 0.05
    }

    
    nx.draw(G, pos, **options)

    plt.axis("off")
    plt.savefig('graph_vis.pdf')


def plot_illus():
    G = nx.Graph()

    G.add_edges_from([(0, 1), (0, 2), (0,3), (0,4), (0,5)])
    node_list = list (G.nodes)
    pos = nx.spring_layout(G)

    options = {
        "nodelist" : node_list,
        "node_size" : 500,
        "node_color": "#ffa600",
        "edge_color": "#66E3D8",
        "width": 3
    }

    nx.draw(G, pos, **options)

    labels={}
    labels[0]='0'
    labels[1]='1'
    labels[2]='2'
    labels[3]='3'
    labels[4]='4'
    labels[5]='5'
    
    nx.draw_networkx_labels(G,pos,labels,font_size=16)

    plt.axis("off")
    plt.savefig('graph_illus.pdf')


def load_mp():
    fname = "datasets/canVote_processed/mp_dict.pkl"
    MP_dict = normal_util.load_object(fname)

    return MP_dict








def export_gephi():

    G_times = canVote_loader.load_canVote_temporarl_edgelist("datasets/canVote_processed/canVote_edgelist.txt")
    MP_dict = load_mp()
    labels = list(range(2006,2020,1))
    print (len(MP_dict))

    for i in range(len(G_times)):
        #party for everyone!
        G = G_times[i]
        count = 0
        for node in G.nodes:
            if (node in MP_dict):
                if (len(MP_dict[node]["party"]) > 0):
                    node_party = MP_dict[node]["party"][-1]
                else:
                    node_party = MP_dict[node]["party"][0]

                if (node_party == 'Conservative'):
                    #blue
                    G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 130, 'b': 189, 'a': 0}}

                # if (node_party == 'Progressive Conservative'):
                #     #Han Purple
                #     #http://www.flatuicolorpicker.com/blue-rgba-color-model/
                #     G.nodes[node]['viz'] = {'color': {'r': 77, 'g': 5, 'b': 232, 'a': 1}}

                # if (node_party == 'Reform'):
                #     #light green
                #     #http://www.flatuicolorpicker.com/green-hex-color-model/
                #     G.nodes[node]['viz'] = {'color': {'r': 123, 'g': 239, 'b': 178, 'a': 1}}

                # if (node_party == 'Canadian Alliance'):
                #     #Mariner
                #     #http://www.flatuicolorpicker.com/blue-rgba-color-model/
                #     G.nodes[node]['viz'] = {'color': {'r': 44, 'g': 130, 'b': 201, 'a': 1}}

                if (node_party == 'Progressive Conservative'):
                    #blue
                    G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 130, 'b': 189, 'a': 0}}

                if (node_party == 'Reform'):
                    #blue
                    G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 130, 'b': 189, 'a': 0}}

                if (node_party == 'Canadian Alliance'):
                    #blue
                    G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 130, 'b': 189, 'a': 0}}

                if (node_party == 'Liberal'):
                    #red
                    G.nodes[node]['viz'] = {'color': {'r': 227, 'g': 74, 'b': 51, 'a': 0}}

                if (node_party == 'Bloc'):
                    #purple
                    G.nodes[node]['viz'] = {'color': {'r': 136, 'g': 86, 'b': 167, 'a': 0}}

                if (node_party == 'NDP'):
                    #green 
                    G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 163, 'b': 84, 'a': 0}}

                if (node_party == 'Independent'):
                    #black
                    G.nodes[node]['viz'] = {'color': {'r': 99, 'g': 99, 'b': 99, 'a': 0}}

                if (node_party == 'Green'):
                    #green 
                    #https://www.greenparty.ca/en/downloads
                    #https://www.color-hex.com/color/3d9b35
                    G.nodes[node]['viz'] = {'color': {'r': 61, 'g': 155, 'b': 53, 'a': 0}}


            else:
                #black is default color
                G.nodes[node]['viz'] = {'color': {'r': 99, 'g': 99, 'b': 99, 'a': 0}}



  #       graph.node['red']['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
        # graph.node['green']['viz'] = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 0}}
        # graph.node['blue']['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}
        # print (count)

        nx.write_gexf(G, "gephi_new/" + str(labels[i]) + ".gexf", version="1.2draft")











def main():
    export_gephi()
    

if __name__ == "__main__":
    main()
