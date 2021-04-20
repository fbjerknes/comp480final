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



def find_next_prime(n):
    prime_cand = n
    while (True):
        for i in range(2, int(np.floor(n ** (1/2))) + 1):
            if (prime_cand % i == 0):
                prime_cand += 1
                break
        return prime_cand

def hash_func(rnge, seed):
    return lambda s: ((mmh3.hash(s.encode('utf-8'),seed=seed)) % find_next_prime(rnge)) % rnge

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
        for j in range(len(A)):
            vals.append(cur(A[j]))
        hashvalues.append(min(vals))
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


def get_candidates(k, l, r, graph, query_node):
    graph_lsh = HashTable(k, l, r)
    hash_funcs = produce_hash(k, l)
    for i in graph.keys():
        codes = generate_hashcodes(graph[i].keys(), k, l, hash_funcs)
        graph_lsh.insert(codes, i)
    query_codes = generate_hashcodes(graph[query_node].keys(), k, l, hash_funcs)
    return graph_lsh.lookup(query_codes)


def create_supernode(graph, query_node, edr_threshold, list_of_candidates):
    query_neighbors = graph[query_node]
    nqsize = len(query_neighbors.keys())

    for node in list_of_candidates:
        other_neighbors = graph[node]
        nvsize = len(other_neighbors.keys())
        potential_compression = query_neighbors | other_neighbors
        nssize = len(potential_compression.keys())
        edr = (nqsize + nvsize - nssize) / (nqsize + nvsize)

        if (edr > edr_threshold):
            graph[query_node] = potential_compression
            del graph[node]


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
            nodes = g.keys()
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g


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
        g[node] = set()

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u + 1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

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
    return result