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
