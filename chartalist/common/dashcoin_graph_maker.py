import networkx as nx
from chartalist.datasets.model.OutPut import OutPut
from chartalist.datasets.model.OutPutAddressModel import OutPutAddressModel
from chartalist.datasets.model.inputs.Input import Input
from chartalist.datasets.model.inputs.InputAddressModel import InputAddressModel


class DashcoinGraphMaker:
    outPutList = list()
    inputList = list()

    # Get and parse out put trans from data file
    def get_out_put(self, df):
        for index, row in df.iterrows():
            Transaction = row['trans'].strip().split('\t')
            newOutPut = OutPut()
            newOutPut.blockHeight = Transaction[0]
            newOutPut.txHash = Transaction[1]
            for i in range(int(Transaction[2])):
                outputAdd = OutPutAddressModel()
                outputAdd.address = Transaction[i + i + 3]
                outputAdd.amount = Transaction[i + i + 4]
                newOutPut.outPuts.append(outputAdd)
            self.outPutList.append(newOutPut)

    # Get and parse in put trans from data file
    def get_input(self, df):
        for index, row in df.iterrows():
            Transaction = row['trans'].strip().split('\t')
            newInput = Input()
            newInput.blockHeight = Transaction[0]
            newInput.txHash = Transaction[1]
            for i in range(int(Transaction[2])):
                inputAdd = InputAddressModel()
                inputAdd.address = Transaction[i + i + 3]
                inputAdd.index = Transaction[i + i + 4]
                newInput.inputs.append(inputAdd)
            self.inputList.append(newInput)

    def make_graph(self, in_df, out_df):
        self.get_input(in_df)
        self.get_out_put(out_df)
        G = nx.DiGraph()
        for i in self.inputList:
            G.add_node(i.txHash, type="trans")
            for i_i in i.inputs:
                G.add_node(i_i.address, type="address")
                G.add_edge(i_i.address, i.txHash)

        for j in self.outPutList:
            G.add_node(j.txHash, type="trans")
            for j_j in j.outPuts:
                G.add_node(j_j.address, type="address")
                G.add_edge(j.txHash, j_j.address)
        return G

    def get_graph_color_map(self, Graph):
        color_map = ['red' if Graph.nodes[node]['type'] == "trans" else 'green' for node in Graph]
        return color_map
