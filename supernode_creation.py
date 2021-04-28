import random
import math
import numpy as np
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
import sys
import mmh3
from collections import defaultdict
from sklearn.utils import murmurhash3_32
import time
import copy






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


def dist(graph_1, graph_2):
    distsum = 0
    for u in graph_1.keys():
        for v in graph_1.keys():
            weight1 = graph_1[u][v]
            weight2 = graph_2[u][v]
            diffsq = (weight2 - weight1) ** 2
            distsum += diffsq
    return distsum

def hash_graph(k, l, r, graph):
    graph_lsh = HashTable(k, l, r)
    hash_funcs = produce_hash(k, l)
    for i in graph.keys():
        codes = generate_hashcodes(graph[i].keys(), k, l, hash_funcs)
        graph_lsh.insert(codes, i)
    return graph_lsh, hash_funcs

def get_candidates(graph, k, l, r, hashed_graph, query_node, hash_funcs):
    query_codes = generate_hashcodes(graph[query_node].keys(), k, l, hash_funcs)
    return hashed_graph.lookup(query_codes)


def create_supernode(graph, query_node, edr_threshold, list_of_candidates):
    if query_node not in graph or not graph[query_node]:
        return
    for node in list_of_candidates:
        query_neighbors = graph[query_node]
        nqsize = len(query_neighbors.keys())
        if node != query_node:
            other_neighbors = graph[node]
            nvsize = len(other_neighbors.keys())
            potential_compression = query_neighbors | other_neighbors
            nssize = len(potential_compression.keys())
            edr = (nqsize + nvsize - nssize) / (nqsize + nvsize)
            # print(edr)
            if (edr > edr_threshold):
                for nbr in graph[node].keys():
                    if nbr == node:
                        continue
                    wgt = graph[node][nbr]
                    if nbr not in graph[query_node].keys() or wgt < graph[query_node][nbr]:
                        graph[nbr][query_node] = wgt
                        graph[query_node][nbr] = wgt
                    del graph[nbr][node]
                del graph[node]


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
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result




g1 = erdos_renyi(20, 0.14)
#print(g1)
g2 = {0: {2: 3, 3: 4, 4: 1}, 1: {2: 2, 3: 7, 4: 1, 5: 4}, 2: {0: 3, 1 : 2}, 3: {0: 4, 1 : 7}, 4: {0: 1, 1 : 1}, 5: {1 : 4}}

k = 2
l = 5
r = 2 ** 12

print(g1)
hg1, hf1 = hash_graph(k, l, r, g1)
for i in range(5):
    if i in g1.keys():
        g_candidates = get_candidates(g1, k, l, r, hg1, i, hf1)
        print(g_candidates)
        create_supernode(g1, i, 0.05, g_candidates)
        print(g1)


g3 = erdos_renyi(10000, 0.005)
g4 = copy.deepcopy(g3)
time1 = time.time()
hg3, hf3 = hash_graph(k, l, r, g3)
for i in range(25):
    if i in g3.keys():
        g3_candidates = get_candidates(g3, k, l, r, hg3, i, hf3)
        create_supernode(g3, i, 0.05, g3_candidates)
time2 = time.time()
print(str(time2 - time1))
time3 = time.time()
for i in range(25):
    if i in g4.keys():
        g4_candidates = list(g4.keys())
        create_supernode(g4, i, 0.05, g4_candidates)
time4 = time.time()
print(str(time4 - time3))