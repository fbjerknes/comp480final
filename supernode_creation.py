import random
import math
import numpy as np
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
import sys
#import mmh3
from collections import defaultdict
from sklearn.utils import murmurhash3_32
import time
import copy
import networkx as nx
import read_graph

NODES_MERGED_THRESHOLD = 200
EDR_THRESHOLD_ERDOS = 0.1
EDR_THRESHOLD_CLUSTERED = 0.2

TIME_SERIES_NODES_MERGED_THRESHOLD = 200

NUM_DATA_POINTS = 30

graph_num = 0


# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self, n):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        plt.figure(n)
        nx.draw_networkx(G)
        plt.show(block=False)

# Driver code


def genhash():
    """
    :return: a random hash function with the values being any unsigned int.
    """
    seed = random.randint(0, 2**32-1)
    return lambda x: murmurhash3_32(x, seed=seed)


def produce_hash(k, l):
    """
    Create l hash tables with k hash functions per table.
    :param k: number of hash functions per table
    :param l: number of hash tables
    :return: the list of hash tables
    """
    h = []

    for i in range(l):
        n = []

        for j in range(k):
            n.append(genhash())

        h.append(n)
    return h


def minhash(A, m, hashes):
    """
    Computes m minhash values of any set
    :param A: Set to be hashed
    :param m: number of hash funcitons -- must be <= len(hashes)
    :param hashes: hash functions to pull from
    :return: A list of minhash values for the set A
    """
    hashvalues = []
    #Iterate over every hash function
    for i in range(m):
        cur = hashes[i]
        vals = []

        for j in A:
            #Append a hash value of every set to a list

            vals.append(cur(j))

        #If there are no hash values, return 0, otherwise return the minimum of the hash values, hence minhash.
        hashvalues.append(0 if len(vals) == 0 else min(vals))

    #return the set of all minhash values for every hash function
    return hashvalues

class HashTable:
    """
    Class for a table to store LSH values
    """
    def __init__(self, K, L, R):
        """
        Create a set of tables of hash values
        :param K: number of hash functions per table
        :param L: number of hash tables
        :param R: range of hash functions
        """
        self.k = K
        self.l = L
        self.r = R
        #hash tables
        self.tables = []

        for i in range(self.l):
            self.tables.append(defaultdict(list))

        #random linear combination constants for hash bucket storage.
        self.a = []

        for i in range(self.k + 1):
            self.a.append(random.randint(0, 2**32 - 1))

    def insert(self, hashcodes, id):
        """
        Stores an item in the hash table
        :param hashcodes: list of lists of hash values
        :param id: ID of item to store
        """
        for i in range(self.l):
            #uses the hashcodes to find a bucket for each table.
            bucket = self.a[-1]

            for j in range(self.k):
                bucket += hashcodes[i][j] * self.a[j]
            bucket = (bucket % (2**31 - 1)) % self.r
            self.tables[i][bucket].append(id)

    def lookup(self, hashcodes):
        """
        Searches through the hash table for a list of hash codes, computes the intersection of all
        values that match at least one of the buckets determined by the hash codes
        :param hashcodes: list of all hashcodes to attempt to match
        :return: values that fit in one or more of the desired buckets
        """
        #find a bucket for every hash table
        bucket = self.a[-1]
        for j in range(self.k):
            bucket += hashcodes[0][j] * self.a[j]
        bucket = (bucket % (2 ** 31 - 1)) % self.r
        idset = set(self.tables[0][bucket])
        #Merge the set of all values in this bucket to the set of IDs to return.
        for i in range(1, self.l):
            bucket = self.a[-1]
            for j in range(self.k):
                bucket += hashcodes[i][j] * self.a[j]
            bucket = (bucket % (2 ** 31 - 1)) % self.r
            idset = idset.union(self.tables[i][bucket])
        return list(idset)


def generate_hashcodes(s, k, l, funcs):
    """
    Hashes an object using the given hash functions
    :param s: the object
    :param k: number of hash functions per table
    :param l: number of hash tables
    :param funcs: all hash fuctions
    :return: table of all hash values
    """
    hashcodes = []
    for i in range(l):
        hashcodes.append(minhash(s, k, funcs[i]))
    return hashcodes


def dist(unmerged, merged, nodes_merged):
    """
    Calculates similarity between a merged and unmerged graph
    :param unmerged: unmerged graph
    :param merged: merged graph
    :param nodes_merged: nodes merged in the graph
    :return: Average value of all Jaccard similarities of neighborhoods of merged nodes in the 2 graphs.
    """

    jaccards = []
    for mergelist in nodes_merged:

        #The supernode is always the first node of the list.
        supernode = mergelist[0]
        merged_nbhd = set(merged[supernode].keys())
        #iterate through every node the supernode was merged with

        for n in mergelist:
            #find the neighborhood of the node in the merged and unmerged graph
            unmerged_nbhd = set(unmerged[n].keys())
            intersection = unmerged_nbhd & merged_nbhd
            union = unmerged_nbhd | merged_nbhd

            #calculate the Jaccard similarity between the merged and unmerged neighborhoods, add to the list
            intersectionsize = (float)(len(intersection))
            jaccard = intersectionsize / len(union)
            jaccards.append(jaccard)

    # return average of all Jaccard values
    return np.average(jaccards)

def hash_graph(k, l, r, graph):
    """
    Hash entire graph into a table of desired size using LSH.
    :param k: number of hash functions per table
    :param l: number of tables
    :param r: range of hash function
    :param graph: the graph to be hashed
    :return: A hash table that contains the hash values of the graph, in addition to the hash functions themselves.
    """

    #creaate hash table and hash functions
    graph_lsh = HashTable(k, l, r)
    hash_funcs = produce_hash(k, l)

    #insert each node of the graph into the hash table
    for i in graph.keys():
        codes = generate_hashcodes(graph[i].keys(), k, l, hash_funcs)
        graph_lsh.insert(codes, i)

    #return the hash table and hash functions
    return graph_lsh, hash_funcs

def get_candidates(graph, k, l, r, hashed_graph, query_node, hash_funcs):
    """
    Using LSH, get the candidates for supernode creation
    :param graph: graph to be compressed
    :param k: number of hash functions per table
    :param l: number of tables
    :param r: range of hash function
    :param hashed_graph: hashed version of the graph
    :param query_node: node to be compressed into
    :param hash_funcs: hash functions to use for the hash table
    :return:
    """
    query_codes = generate_hashcodes(graph[query_node].keys(), k, l, hash_funcs)
    return hashed_graph.lookup(query_codes)


def create_supernode(graph, query_node, edr_threshold, list_of_candidates, cur_nodes_merged):
    """
    Using a set of candidates, merge nodes that satisfy certain criteria into a supernode
    :param graph: graph to be compressed
    :param query_node: will become the supernode
    :param edr_threshold: minimum EDR value for merging
    :param list_of_candidates: possible nodes to merge into the supernode
    :param cur_nodes_merged: List of nodes currently merged, to be added to with the current supernode
    :return: number of nodes merged into the supernode
    """

    nodes_merged = 0
    #visualize the graph if it's small enough
    vis = len(graph.keys()) < 20

    #you can't merge if the query node isn't in the graph.
    if query_node not in graph or not graph[query_node]:
        return 0

    #test each node in the list of candidates
    for node in list_of_candidates:

        #calculate the neighbors of the query node
        query_neighbors = graph[query_node]

        #number of neighbors of query node
        nqsize = len(query_neighbors.keys())

        #if the candidate is still in the graph and it isn't the query node, you can consider it for merging.
        if node in graph and node != query_node:
            other_neighbors = graph[node]
            nvsize = len(other_neighbors.keys())

            #we want the size of a potential merging of the two nodes
            potential_compression = query_neighbors | other_neighbors
            nssize = len(potential_compression.keys())

            #EDR value will tell us if we merge this node
            edr = (nqsize + nvsize - nssize) / (nqsize + nvsize)

            #We don't want to merge nodes that don't have enough neighbors compared to the query
            sizedif = nqsize / nssize
            if (sizedif < 1 and sizedif != 0):
                sizedif = 1/sizedif

            if (edr > edr_threshold and sizedif < 6):
                #MERGING PROCESS
                nodes_merged += 1

                for nbr in graph[node].keys():
                    #We want to remove references to the node to be merged in its neighbors
                    if nbr == node:
                        continue
                    wgt = graph[node][nbr]

                    #Add the neighbor to the supernode and keep the maximum weight.
                    if nbr not in graph[query_node].keys() or wgt < graph[query_node][nbr]:
                        graph[nbr][query_node] = wgt
                        graph[query_node][nbr] = wgt

                    #delete the reference to the merged nodode
                    del graph[nbr][node]

                #complete the compression
                del graph[node]

                #Add the new node and its supernode to the list of merged node lists.
                mergequery = False
                mergenode = False

                #find if the supernode and the merged node are already in the merged list.
                for i in range(len(cur_nodes_merged)):
                    if query_node in cur_nodes_merged[i]:
                        mergequery = True
                        querylist = i
                    if node == cur_nodes_merged[i][0]:
                        mergenode = True
                        nodelist = i
                    if mergenode and mergequery:
                        break

                #If neither node is in the merged list, create a new entry.
                if not mergenode and not mergequery:
                    cur_nodes_merged.append([query_node, node])


                elif mergequery:
                    if mergenode:

                        #If both nodes are in the merged list, append the merged node's list to the supernode's list
                        #and delete the merged node.
                        cur_nodes_merged[querylist].extend(cur_nodes_merged[nodelist])
                        cur_nodes_merged.pop(nodelist)

                    else:
                        #If just the supernode is in the merged list, add the new merged node to its list.
                        cur_nodes_merged[querylist].append(node)
                else:
                    #If just the merged node is in the merged list, it becomes the supernode's list.
                    cur_nodes_merged[nodelist].insert(0, query_node)

                    #If the graph is small, visualize it.
                if vis:
                    make_visual(graph)
    return nodes_merged

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = {}

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u + 1, n):
            r = random.random()
            if r < p:
                w = random.randint(1, 10)
                g[u][v] = w
                g[v][u] = w

    return g


def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))


def random_graph_generator(n):
    """
    Generates a semi-random clustered graph
    :param n: number of nodes in the graph
    :return: the random graph.
    """
    rg = defaultdict(lambda: defaultdict(int))

    #If the nodes are within the same range of 10, the probability of an edge between them is 0.9.
    #Otherwise, the probability of an edge between them is 0.001.
    for i in range(n):
        for j in range(i):
            ran = random.random()
            if int(i//10) == int(j//10):
                if ran < 0.9:
                    rg[i][j] = 1
                    rg[j][i] = 1
            else:
                if ran < 0.001:
                    rg[i][j] = 1
                    rg[j][i] = 1
    for i in rg.keys():
        rg[i] = dict(rg[i])
    return dict(rg)



g1 = erdos_renyi(20, 0.14)
g2 = {0: {2: 3, 3: 4, 4: 1}, 1: {2: 2, 3: 7, 4: 1, 5: 4}, 2: {0: 3, 1 : 2}, 3: {0: 4, 1 : 7}, 4: {0: 1, 1 : 1}, 5: {1 : 4}}

k = 2
l = 5
r = 2 ** 12



def make_visual(graph):
    """
    Visualize a graph
    :param graph: graph to be visualized.
    """
    global graph_num
    V = GraphVisualization()
    for node in graph.keys():
        for nbr in g9[node].keys():
            V.addEdge(node, nbr)
    V.visualize(graph_num)
#     graph_num += 1

#
# ###VISUALIZE A SMALL GRAPH AND PERFORM VERY FEW COMPRESS
# g9 = erdos_renyi(10, 0.3)
# hg9, hf9 = hash_graph(k, l, r, g9)
# make_visual(g9)
# nm9 = 0
# g9_merges = []
# for i in range(len(g9)):
#     if i in g9.keys():
#         g9_candidates = g9_candidates = get_candidates(g9, k, l, r, hg9, i, hf9)
#         nm9 += create_supernode(g9, i, EDR_THRESHOLD_ERDOS, g9_candidates, g9_merges)
#     if nm9 > 3:
#         break
#
#
#
# plt.show()


def run_tests():
    ### ERDOS RENYI TEST
    print("REAL WORLD")
    g3 = read_graph.read_graph("USAir97.mtx")
    g = open("graph_out.txt", "w")
    g.write(str(g3))
    #print(g3)
    # timeE2 = time.time()
    # print(str(timeE2 - timeE1))

    ###FOR NON-LSH ALGORITHM
    g4 = copy.deepcopy(g3)
    ###ORIGINAL UNMERGED GRAPH
    g5 = copy.deepcopy(g3)

    ###LSH AlGORITHM TIME TEST
    time1 = time.time()
    #Hash the graph
    hg3, hf3 = hash_graph(k, l, r, g3)
    nodes_merged = 0
    g3_merges = []
    #Merge a given number of nodes.
    for i in range(len(g3)):
        if i in g3.keys():
            g3_candidates = get_candidates(g3, k, l, r, hg3, i, hf3)
            nodes_merged += create_supernode(g3, i, EDR_THRESHOLD_ERDOS, g3_candidates, g3_merges)
        if nodes_merged > NODES_MERGED_THRESHOLD:
            break


    print("Nodes Merged with LSH: " + str(nodes_merged))
    time2 = time.time()

    print("Time for merging nodes with LSH: " + str(time2 - time1))

    ###NON-LSH TIME TEST
    time3 = time.time()
    nm = 0
    g4_merges = []

    for i in range(len(g4)):
        if i in g4.keys():
            g4_candidates = list(g4.keys())
            nm += create_supernode(g4, i, EDR_THRESHOLD_ERDOS, g4_candidates, g4_merges)
        if nm > nodes_merged:
            break


    print("Nodes Merged without LSH: " + str(nm))
    time4 = time.time()
    print("Time for merging nodes without LSH: " + str(time4 - time3))

    ###Use the similalrity function to see how similar the two merged graphs are to the original graph

    print("Closeness of Merged Sets, with LSH: " + str(dist(g5, g3, g3_merges)))
    print("Closeness of Merged Sets, without LSH: " + str(dist(g5, g4, g4_merges)))
    print("")
    print("")


    print("CLUSTERED TEST")

    ###SAME TIME AND ACCURACY TESTS AS PREVIOUS, just with the clustered graph.
    g6 = random_graph_generator(10000)
    g7 = copy.deepcopy(g6)
    g8 = copy.deepcopy(g6)
    time1 = time.time()
    hg6, hf6 = hash_graph(k, l, r, g6)
    nodes_merged = 0
    g6_merges = []


    for i in range(len(g6)):
        if i in g6.keys():
            g6_candidates = get_candidates(g6, k, l, r, hg6, i, hf6)
            nodes_merged += create_supernode(g6, i, EDR_THRESHOLD_CLUSTERED, g6_candidates, g6_merges)
        if nodes_merged > NODES_MERGED_THRESHOLD:
            break


    print("Nodes Merged with LSH: " + str(nodes_merged))
    time2 = time.time()
    print("Time for merging nodes with LSH: " + str(time2 - time1))
    time3 = time.time()
    nm = 0

    g7_merges = []
    for i in range(len(g7)):
        if i in g7.keys():
            g7_candidates = list(g7.keys())
            nm += create_supernode(g7, i, EDR_THRESHOLD_CLUSTERED, g7_candidates, g7_merges)
        if nm > nodes_merged:
            break


    print("Nodes Merged without LSH: " + str(nm))
    time4 = time.time()
    print("Time for merging nodes without LSH: " + str(time4 - time3))
    print("Closeness of Merged Sets, with LSH: " + str(dist(g8, g6, g6_merges)))
    print("Closeness of Merged Sets, without LSH: " + str(dist(g8, g7, g7_merges)))


def produce_time_series():
    """
    Compare the simlarity scores of the LSH vs non-LSH methods after each iteration of node-merging
    """
    g3 = erdos_renyi(10000, 0.001)

    ###FOR NON-LSH ALGORITHM
    g4 = copy.deepcopy(g3)
    ###ORIGINAL UNMERGED GRAPH
    g5 = copy.deepcopy(g3)

    hg3, hf3 = hash_graph(k, l, r, g3)
    nodes_merged = 0
    g3_merges = []

    #Similarity values
    er_simi = []

    ###Compute similarity values using LSH
    for i in range(len(g3)):
        if i in g3.keys():
            g3_candidates = get_candidates(g3, k, l, r, hg3, i, hf3)
            c = create_supernode(g3, i, EDR_THRESHOLD_ERDOS, g3_candidates, g3_merges)
            nodes_merged += c
            for j in range(c):
                er_simi.append(dist(g5, g3, g3_merges))
        if nodes_merged > TIME_SERIES_NODES_MERGED_THRESHOLD:
            break

    nm = 0
    g4_merges = []
    bf_simi = []

    ###Compute similarity values using non-LSH method.
    for i in range(len(g4)):
        if i in g4.keys():
            g4_candidates = list(g4.keys())
            d = create_supernode(g4, i, EDR_THRESHOLD_ERDOS, g4_candidates, g4_merges)
            nm += d
            for j in range(d):
                bf_simi.append(dist(g5, g4, g4_merges))
        if nm > nodes_merged:
            break

    m = min(len(er_simi), len(bf_simi))
    return {'Nodes Removed': range(nodes_merged), 'With LSH': er_simi[:m], 'Without LSH': bf_simi[:m]}


def produce_plot_data():
    """
    Plot all desired data
    """

    ###ERDOS RENYI GRAPH PLOTS
    er_data = [[[], [], []], [[], [], []]]
    for a in range(NUM_DATA_POINTS):
        print("a = " + str(a))
        g3 = erdos_renyi(10000, 0.001)


        ###FOR NON-LSH ALGORITHM
        g4 = copy.deepcopy(g3)
        ###ORIGINAL UNMERGED GRAPH
        g5 = copy.deepcopy(g3)

        time1 = time.time()
        hg3, hf3 = hash_graph(k, l, r, g3)
        nodes_merged = 0
        g3_merges = []

        for i in range(len(g3)):
            if i in g3.keys():
                g3_candidates = get_candidates(g3, k, l, r, hg3, i, hf3)
                nodes_merged += create_supernode(g3, i, EDR_THRESHOLD_ERDOS, g3_candidates, g3_merges)
            if nodes_merged > NODES_MERGED_THRESHOLD:
                break

        er_data[0][0].append(nodes_merged)
        time2 = time.time()
        er_data[0][1].append(time2 - time1)
        time3 = time.time()

        nm = 0
        g4_merges = []

        for i in range(len(g4)):
            if i in g4.keys():
                g4_candidates = list(g4.keys())
                nm += create_supernode(g4, i, EDR_THRESHOLD_ERDOS, g4_candidates, g4_merges)
            if nm > nodes_merged:
                break

        er_data[1][0].append(nm)
        time4 = time.time()
        er_data[1][1].append(time4 - time3)
        er_data[0][2].append(dist(g5, g3, g3_merges))
        er_data[1][2].append(dist(g5, g4, g4_merges))

    time_dif = []
    sim_dif = []
    for i in range(len(er_data[0][1])):
        time_dif.append(float(er_data[1][1][i]) / er_data[0][1][i])
        sim_dif.append(float(er_data[0][2][i]) / er_data[1][2][i])

    er_plot_time = {'Without LSH': er_data[1][1], 'With LSH': er_data[0][1]}
    er_plot_similarity = {'Without LSH': er_data[1][2], 'With LSH': er_data[0][2]}
    er_plot_differences = {'Time Difference': time_dif, 'Accuracy Difference': sim_dif}

    ###CLUSTERED GRAPH PLOTS
    rand_data = [[[], [], []], [[], [], []]]

    for a in range(NUM_DATA_POINTS):
        print("a = " + str(a))
        g9 = random_graph_generator(10000)
        g10 = copy.deepcopy(g9)
        g11 = copy.deepcopy(g9)
        time1 = time.time()
        hg9, hf9 = hash_graph(k, l, r, g9)
        nodes_merged = 0
        g9_merges = []

        for i in range(len(g9)):
            if i in g9.keys():
                g9_candidates = get_candidates(g9, k, l, r, hg9, i, hf9)
                nodes_merged += create_supernode(g9, i, EDR_THRESHOLD_CLUSTERED, g9_candidates, g9_merges)
            if nodes_merged > NODES_MERGED_THRESHOLD:
                break

        rand_data[0][0].append(nodes_merged)
        time2 = time.time()
        rand_data[0][1].append(time2 - time1)
        time3 = time.time()
        nm = 0

        g10_merges = []
        for i in range(len(g10)):
            if i in g10.keys():
                g10_candidates = list(g10.keys())
                nm += create_supernode(g10, i, EDR_THRESHOLD_CLUSTERED, g10_candidates, g10_merges)
            if nm > nodes_merged:
                break

        rand_data[1][0].append(nm)
        time4 = time.time()
        rand_data[1][1].append(time4 - time3)
        rand_data[0][2].append(dist(g11, g9, g9_merges))
        rand_data[1][2].append(dist(g11, g10, g10_merges))

    time_dif = []
    sim_dif = []
    for i in range(len(rand_data[0][1])):
        time_dif.append(float(rand_data[1][1][i]) / rand_data[0][1][i])
        sim_dif.append(float(rand_data[0][2][i]) / rand_data[1][2][i])

    rand_plot_time = {'Without LSH': rand_data[1][1], 'With LSH': rand_data[0][1]}
    rand_plot_similarity = {'Without LSH': rand_data[1][2], 'With LSH': rand_data[0][2]}
    rand_plot_differences = {'Time Difference': time_dif, 'Accuracy Difference': sim_dif}

    return er_plot_time, er_plot_similarity, er_plot_differences, rand_plot_time, rand_plot_similarity, rand_plot_differences

run_tests()

series = produce_time_series()
data = produce_plot_data()

actual = pd.DataFrame(data=series)

#Plot the time series
actual.plot(x='Nodes Removed')
plt.show()

#For both the Erdos-Reyni and the clustered graph, plot the following comparisons with the non-LSH algorithm:
#1. Time elapsed of LSH vs. Time Elapsed of non-LSH
#2. Accuracy of LSH vs. Accuracy of non-LSH
#3. Ratio of LSH-to-non-LSH time elapsed vs Ratio of LSH-to-non-LSH accuracy.

for i in range(6):
    graph_num += 1
    fig = plt.figure(graph_num)
    if i % 3 == 2:
        plt.scatter(x=data[i]['Time Difference'], y=data[i]['Accuracy Difference'])
        ax = fig.add_subplot()
        ax.set_xlabel('Times Faster with LSH (Time of Brute Force/Time of LSH)')
        ax.set_ylabel('Accuracy Ratio (Similarity With LSH/With Brute Force')
    else:
        plt.scatter(x=data[i]['Without LSH'], y=data[i]['With LSH'])
        ax = fig.add_subplot()
        ax.set_xlabel('Without LSH')
        ax.set_ylabel('With LSH')


plt.show()