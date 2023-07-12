import dateutil.parser as dparser
import re
import networkx as nx
import pickle

def load_object(filename):
    output = 0
    with open(filename, 'rb') as fp:
        output = pickle.load(fp)
    return output


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 2)


def pkl2edgelist(fname, outName):
    temp_dict = load_object(fname)
    outfile = open(outName, "w")
    temps = list(temp_dict.keys())
    temps.sort()
    print (temps)
    for i in range(len(temps)):
        # (u,v) : w
        edges = {}
        for (u,v) in temp_dict[temps[i]]:
            if ((u,v) in edges):
                edges[(u,v)] = edges[(u,v)] + 1
            else:
                edges[(u,v)] = 1

        for (u,v) in edges:
            outfile.write(str(temps[i]) + "," + str(u) + "," + str(v) + "," + str(edges[(u,v)]) + "\n")
    outfile.close()









'''
treat each year as a timestamp 
'''
def load_canVote_temporarl_edgelist(fname):
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
    #idx = 0   #gephi

    for i in range(0, len(lines)):
        line = lines[i]
        values = line.split(',')
        t = values[0]
        v = values[1]       
        u = values[2]
        w = int(values[3])  #edge weight by number of shared publications in a year
        if current_date != '':
            if t != current_date:
                G_times.append(G)   #append old graph
                #nx.write_gexf(G, str(idx+2006) + ".gexf")      #gephi
                #idx = idx + 1      #gephi
                G = nx.DiGraph()    #create new graph
                current_date = t
        else:
            current_date = t
        G.add_edge(u, v, weight=w)
    #don't forget to add last one!!!!
    G_times.append(G)
    #nx.write_gexf(G, str(idx+2006) + ".gexf")      #gephi

    print ("maximum time stamp is " + str(len(G_times)))
    return G_times


def load_csv():
    MP_dict = {}
    fname = "party_politics.csv"
    file = open(fname, "r")
    file.readline()
    for line in file.readlines():
        line = line.strip("\n")
        values = line.split(",")
        u = values[-2]
        party = values[-1]
        MP_dict[u] = party
    return MP_dict


def load_pkl():
    MP_dict=load_object("mp_dict.pkl")
    return MP_dict



def main():
    # G_times = load_canVote_temporarl_edgelist("canVote_processed/canVote_edgelist.txt")

    # MP_dict = load_pkl()
    # print (len(MP_dict))
    # labels = list(range(2006,2020,1))

    # parties = []
    # for key in MP_dict.keys():
    #     if MP_dict[key] not in parties:
    #         parties.append(MP_dict[key])

    # # print (MP_dict)

    # noParties = {}
    # for G in G_times:
    #     for n in G.nodes():
    #         if n not in MP_dict and n not in noParties:
    #             noParties[n] = 1

    # print (MP_dict)

    pkl2edgelist("temp_edgelist.pkl", "canVote_edgelist.txt")


    


if __name__ == "__main__":
    main()
