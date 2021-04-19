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
import pandas as pd
import time
import numpy as np


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
        for j in range(len(A) - 2):
            vals.append(cur(A[j:j+3]))
        hashvalues.append(min(vals))
    return hashvalues


def compare_strings(x, y):
    hundred = []
    for i in range(100):
        hundred.append(genhash())
    hashx = minhash(x, 100, hundred)
    hashy = minhash(y, 100, hundred)
    count = 0
    for i in range(100):
        if hashx[i] == hashy[i]:
            count += 1
    return count / 100


s1 = "The mission statement of the WCSCC and area employers recognize the importance of good attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced placement as well as hindering his/her likelihood for successfully completing their program."
s2 = "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. Any student who is absent more than 18 days will loose the opportunity for successfully completing their trade program."

print(compare_strings(s1, s2))


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
