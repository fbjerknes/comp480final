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

NODES_MERGED_THRESHOLD = 100
EDR_THRESHOLD = 0.1

TIME_SERIES_NODES_MERGED_THRESHOLD = 30

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

def minhash(set_a, set_b, hashes, seed):
    hashedset = []
    for elem in set_a:
        hashedset.append(hash_func(2**32, seed)(elem))
    return min(hashedset)

class SuperNodeCreator:
    def __init__(self, graph, num_hashes, compression_factor):
        self.graph = graph
        self.num_hashes = num_hashes
        self.compression_factor = compression_factor
    def insert_node(self, node):
        pass
    def delete_node(self, node):
        pass
    def nodes(self):
        pass
    def super_nodes(self):
        pass
    def edges(self):
        pass
    def super_edges(self):
        pass
    def get_neighbors(self, node):
        pass
    def get_supernode(self, node):
        pass
    def get_subnodes(self, supernode):
        pass

def lsh(set_a):
    return 0


def compress(graph):
    newgraph = {}
    for i in graph.keys():
        newgraph[i] = lsh(graph[i])


def genhash():
    seed = random.randint(0, 2**32-1)
    return lambda x: murmurhash3_32(x, seed=seed)


def produce_hash(k, l):
    h = []
    for i in range(l):
        n = []
        for j in range(k):
            n.append(genhash())
        h.append(n)
    return h


def minhash(A, m, hashes):
    hashvalues = []
    for i in range(m):
        cur = hashes[i]
        vals = []
        for j in A:
            vals.append(cur(j))
        hashvalues.append(0 if len(vals) == 0 else min(vals))
    return hashvalues



class HashTable:
    def __init__(self, K, L, R):
        self.k = K
        self.l = L
        self.r = R
        self.tables = []
        for i in range(self.l):
            self.tables.append(defaultdict(list))
        self.a = []
        for i in range(self.k + 1):
            self.a.append(random.randint(0, 2**32 - 1))

    def insert(self, hashcodes, id):
        for i in range(self.l):
            bucket = self.a[-1]
            for j in range(self.k):
                bucket += hashcodes[i][j] * self.a[j]
            bucket = (bucket % (2**31 - 1)) % self.r
            self.tables[i][bucket].append(id)

    def lookup(self, hashcodes):
        bucket = self.a[-1]
        for j in range(self.k):
            bucket += hashcodes[0][j] * self.a[j]
        bucket = (bucket % (2 ** 31 - 1)) % self.r
        idset = set(self.tables[0][bucket])
        for i in range(1, self.l):
            bucket = self.a[-1]
            for j in range(self.k):
                bucket += hashcodes[i][j] * self.a[j]
            bucket = (bucket % (2 ** 31 - 1)) % self.r
            idset = idset.union(self.tables[i][bucket])
        return list(idset)


def generate_hashcodes(s, k, l, funcs):
    hashcodes = []
    for i in range(l):
        hashcodes.append(minhash(s, k, funcs[i]))
    return hashcodes


def dist(unmerged, merged, nodes_merged):
    jaccards = []
    for mergelist in nodes_merged:
        supernode = mergelist[0]
        merged_nbhd = set(merged[supernode].keys())
        for n in mergelist:
            unmerged_nbhd = set(unmerged[n].keys())
            intersection = unmerged_nbhd & merged_nbhd
            union = unmerged_nbhd | merged_nbhd
            intersectionsize = (float)(len(intersection))
            jaccard = intersectionsize / len(union)
            jaccards.append(jaccard)
    return np.average(jaccards)

def hash_graph(k, l, r, graph):
    graph_lsh = HashTable(k, l, r)
    hash_funcs = produce_hash(k, l)
    for i in graph.keys():
        codes = generate_hashcodes(graph[i].keys(), k, l, hash_funcs)
        graph_lsh.insert(codes, i)
    return graph_lsh, hash_funcs

def get_candidates(graph, k, l, r, hashed_graph, query_node, hash_funcs):
    query_codes = generate_hashcodes(graph[query_node].keys(), k, l, hash_funcs)
    candidates = hashed_graph.lookup(query_codes)
    return hashed_graph.lookup(query_codes)


def create_supernode(graph, query_node, edr_threshold, list_of_candidates, cur_nodes_merged):
    nodes_merged = 0
    vis = len(graph.keys()) < 20
    if query_node not in graph or not graph[query_node]:
        return 0
    for node in list_of_candidates:
        query_neighbors = graph[query_node]
        nqsize = len(query_neighbors.keys())
        if node in graph and node != query_node:
            other_neighbors = graph[node]
            nvsize = len(other_neighbors.keys())
            potential_compression = query_neighbors | other_neighbors
            nssize = len(potential_compression.keys())
            edr = (nqsize + nvsize - nssize) / (nqsize + nvsize)
            sizedif = nqsize / nssize
            if (sizedif < 1 and sizedif != 0):
                sizedif = 1/sizedif
            # print(edr)
            if (edr > edr_threshold and sizedif < 6):
                nodes_merged += 1
                for nbr in graph[node].keys():
                    if nbr == node:
                        continue
                    wgt = graph[node][nbr]
                    if nbr not in graph[query_node].keys() or wgt < graph[query_node][nbr]:
                        graph[nbr][query_node] = wgt
                        graph[query_node][nbr] = wgt
                    del graph[nbr][node]
                del graph[node]
                mergequery = False
                mergenode = False
                for i in range(len(cur_nodes_merged)):
                    if query_node in cur_nodes_merged[i]:
                        mergequery = True
                        querylist = i
                    if node == cur_nodes_merged[i][0]:
                        mergenode = True
                        nodelist = i
                    if mergenode and mergequery:
                        break
                if not mergenode and not mergequery:
                    cur_nodes_merged.append([query_node, node])
                elif mergequery:
                    if mergenode:
                        cur_nodes_merged[querylist].extend(cur_nodes_merged[nodelist])
                        cur_nodes_merged.pop(nodelist)
                    else:
                        cur_nodes_merged[querylist].append(node)
                else:
                    cur_nodes_merged[nodelist].insert(0, query_node)
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


def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = {}
            for idx in mult.keys():
                node = nodes[idx]
                g[new_node][node] = 1
                g[node][new_node] = 1
    return g


def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.

    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.

    Arguments:
    num_nodes -- The number of nodes in the returned graph.

    Returns:
    A complete graph in dictionary form.
    """
    result = {}

    for node_key in range(num_nodes):
        result[node_key] = {}
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key][node_value] = 1

    return result


def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = np.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    r = {}
    for i in result:
        r[i] = 1
    return r



def random_graph_generator(n):
    rg = defaultdict(lambda: defaultdict(int))
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
#print(g1)
g2 = {0: {2: 3, 3: 4, 4: 1}, 1: {2: 2, 3: 7, 4: 1, 5: 4}, 2: {0: 3, 1 : 2}, 3: {0: 4, 1 : 7}, 4: {0: 1, 1 : 1}, 5: {1 : 4}}

k = 2
l = 5
r = 2 ** 12



def make_visual(graph):
    global graph_num
    V = GraphVisualization()
    for node in graph.keys():
        for nbr in g9[node].keys():
            V.addEdge(node, nbr)
    V.visualize(graph_num)
    graph_num += 1

# timeE1 = time.time()
print("VISUALIZING A GRAPH")
g9 = erdos_renyi(10, 0.3)
hg9, hf9 = hash_graph(k, l, r, g9)
make_visual(g9)
nm9 = 0
g9_merges = []
for i in range(len(g9)):
    if i in g9.keys():
        g9_candidates = g9_candidates = get_candidates(g9, k, l, r, hg9, i, hf9)
        nm9 += create_supernode(g9, i, EDR_THRESHOLD, g9_candidates, g9_merges)
    if nm9 > 3:
        break



plt.show()

def run_tests():
    ### ERDOS RENYI TEST
    print("ERDOS REYNI")
    g3 = erdos_renyi(10000, 0.001)

    #print(g3)
    # timeE2 = time.time()
    # print(str(timeE2 - timeE1))

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
            nodes_merged += create_supernode(g3, i, EDR_THRESHOLD, g3_candidates, g3_merges)
        if nodes_merged > NODES_MERGED_THRESHOLD:
            break


    print("Nodes Merged with LSH: " + str(nodes_merged))
    time2 = time.time()
    print("Time for merging nodes with LSH: " + str(time2 - time1))
    time3 = time.time()


    nm = 0
    g4_merges = []

    for i in range(len(g4)):
        if i in g4.keys():
            g4_candidates = list(g4.keys())
            nm += create_supernode(g4, i, EDR_THRESHOLD, g4_candidates, g4_merges)
        if nm > nodes_merged:
            break


    print("Nodes Merged without LSH: " + str(nm))
    time4 = time.time()
    print("Time for merging nodes without LSH: " + str(time4 - time3))
    print("Closeness of Merged Sets, with LSH: " + str(dist(g5, g3, g3_merges)))
    print("Closeness of Merged Sets, without LSH: " + str(dist(g5, g4, g4_merges)))
    print("")
    print("")


    print("UPA TEST")

    g6 = upa(10000, 100)
    g7 = copy.deepcopy(g6)
    g8 = copy.deepcopy(g6)
    time1 = time.time()
    hg6, hf6 = hash_graph(k, l, r, g6)
    nodes_merged = 0
    g6_merges = []


    for i in range(len(g6)):
        if i in g6.keys():
            g6_candidates = get_candidates(g6, k, l, r, hg6, i, hf6)
            nodes_merged += create_supernode(g6, i, EDR_THRESHOLD, g6_candidates, g6_merges)
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
            nm += create_supernode(g7, i, EDR_THRESHOLD, g7_candidates, g7_merges)
        if nm > nodes_merged:
            break


    print("Nodes Merged without LSH: " + str(nm))
    time4 = time.time()
    print("Time for merging nodes without LSH: " + str(time4 - time3))
    print("Closeness of Merged Sets, with LSH: " + str(dist(g8, g6, g6_merges)))
    print("Closeness of Merged Sets, without LSH: " + str(dist(g8, g7, g7_merges)))


def produce_time_series():
    g3 = erdos_renyi(10000, 0.001)

    # print(g3)
    # timeE2 = time.time()
    # print(str(timeE2 - timeE1))

    ###FOR NON-LSH ALGORITHM
    g4 = copy.deepcopy(g3)
    ###ORIGINAL UNMERGED GRAPH
    g5 = copy.deepcopy(g3)

    hg3, hf3 = hash_graph(k, l, r, g3)
    nodes_merged = 0
    g3_merges = []

    er_simi = []

    for i in range(len(g3)):
        if i in g3.keys():
            g3_candidates = get_candidates(g3, k, l, r, hg3, i, hf3)
            c = create_supernode(g3, i, EDR_THRESHOLD, g3_candidates, g3_merges)
            nodes_merged += c
            for j in range(c):
                er_simi.append(dist(g5, g3, g3_merges))
        if nodes_merged > TIME_SERIES_NODES_MERGED_THRESHOLD:
            break

    nm = 0
    g4_merges = []
    bf_simi = []

    for i in range(len(g4)):
        if i in g4.keys():
            g4_candidates = list(g4.keys())
            d = create_supernode(g4, i, EDR_THRESHOLD, g4_candidates, g4_merges)
            nm += d
            for j in range(d):
                bf_simi.append(dist(g5, g4, g4_merges))
        if nm > nodes_merged:
            break

    return {'Nodes Removed': range(nodes_merged), 'With LSH': er_simi, 'Without LSH': bf_simi}


def produce_plot_data():
    er_data = [[[], [], []], [[], [], []]]
    for a in range(NUM_DATA_POINTS):
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
                nodes_merged += create_supernode(g3, i, EDR_THRESHOLD, g3_candidates, g3_merges)
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
                nm += create_supernode(g4, i, EDR_THRESHOLD, g4_candidates, g4_merges)
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


    upa_data = [[[], [], []], [[], [], []]]

    g6 = upa(10000, 100)
    g7 = copy.deepcopy(g6)
    g8 = copy.deepcopy(g6)
    time1 = time.time()
    hg6, hf6 = hash_graph(k, l, r, g6)
    nodes_merged = 0
    g6_merges = []

    for i in range(len(g6)):
        if i in g6.keys():
            g6_candidates = get_candidates(g6, k, l, r, hg6, i, hf6)
            nodes_merged += create_supernode(g6, i, EDR_THRESHOLD, g6_candidates, g6_merges)
        if nodes_merged > NODES_MERGED_THRESHOLD:
            break

    upa_data[0][0].append(nodes_merged)
    time2 = time.time()
    upa_data[0][1].append(time2 - time1)
    time3 = time.time()
    nm = 0

    g7_merges = []
    for i in range(len(g7)):
        if i in g7.keys():
            g7_candidates = list(g7.keys())
            nm += create_supernode(g7, i, EDR_THRESHOLD, g7_candidates, g7_merges)
        if nm > nodes_merged:
            break

    upa_data[1][0].append(nm)
    time4 = time.time()
    upa_data[1][1].append(time4 - time3)
    upa_data[0][2].append(dist(g8, g6, g6_merges))
    upa_data[1][2].append(dist(g8, g7, g7_merges))

    time_dif = []
    sim_dif = []
    for i in range(len(upa_data[0][1])):
        time_dif.append(float(upa_data[1][1][i]) / upa_data[0][1][i])
        sim_dif.append(float(upa_data[0][2][i]) / upa_data[1][2][i])

    upa_plot_time = {'Without LSH': upa_data[1][1], 'With LSH': upa_data[0][1]}
    upa_plot_similarity = {'Without LSH': upa_data[1][2], 'With LSH': upa_data[0][2]}
    upa_plot_differences = {'Time Difference': time_dif, 'Accuracy Difference': sim_dif}


    rand_data = [[[], [], []], [[], [], []]]

    g9 = upa(10000, 100)
    g10 = copy.deepcopy(g9)
    g11 = copy.deepcopy(g9)
    time1 = time.time()
    hg9, hf9 = hash_graph(k, l, r, g9)
    nodes_merged = 0
    g9_merges = []

    for i in range(len(g9)):
        if i in g9.keys():
            g9_candidates = get_candidates(g9, k, l, r, hg9, i, hf9)
            nodes_merged += create_supernode(g9, i, EDR_THRESHOLD, g9_candidates, g9_merges)
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
            nm += create_supernode(g10, i, EDR_THRESHOLD, g10_candidates, g10_merges)
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

    return er_plot_time, er_plot_similarity, er_plot_differences, upa_plot_time, upa_plot_similarity, upa_plot_differences, rand_plot_time, rand_plot_similarity, rand_plot_differences


series = produce_time_series()
data = produce_plot_data()

actual = [pd.DataFrame(data=series)]

for i in data:
    actual.append(pd.DataFrame(data=i))

for i in range(10):
    if i == 0:
        actual[i].plot(x='Nodes Removed')
    elif i % 3 == 0:
        actual[i].plot(x='Time Difference')
    else:
        actual[i].plot(x='Without LSH')

plt.show()