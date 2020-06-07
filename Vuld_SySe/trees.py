import numpy as np


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', False):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def height(self):
        if getattr(self, '_height', False):
            return self._height
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_height = self.children[i].height()
                if child_height > count:
                    count = child_height
            count += 1
        self._height = count
        return self._height

    def depth(self):
        if getattr(self, '_depth', False):
            return self._depth
        count = 0
        if self.parent:
            count += self.parent.depth()
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

    def set_idx(self, idx):
        self.idx = idx

    def pretty_string(self, tab=0):
        repr = ''
        for _ in range(tab):
            repr += '\t'
        repr += (str(self.idx) + '\n')
        for c in self.children:
            repr += c.pretty_string(tab+1)
        return repr


def set_tree_indices(tree):
    idx = 0
    queue = [tree]
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        t.idx = idx
        idx += 1
        queue += t.children


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1
    return ret


def tree_to_dist_mat(tree, directed=True):
    l = tree.size()
    adj_matrix = tree_to_adj(l, tree, directed=directed, self_loop=False)
    dist_mat = np.array([[float('inf')] * l] *l)
    for i in range(l):
        for j in range(l):
            if i == j:
                dist_mat[i, j] = 0.
            elif adj_matrix[i, j] == 1:
                dist_mat[i, j] = 1.

    for k in range(l):
        for i in range(l):
            for j in range(l):
                d = dist_mat[i, k] + dist_mat[k, j]
                if d < dist_mat[i, j]:
                    dist_mat[i, j] = d
    return dist_mat

def json_to_tree(json_dict):
    value = json_dict["v"]
    children = json_dict["c"]
    tree = Tree(value)
    for child in children:
        tree.add_child(json_to_tree(child))
    return tree


if __name__ == '__main__':
    tree_json = {
        'v': 0,
        'c': [
            {
                'v': 1,
                'c': [
                    {
                        'v': 2,
                        'c': []
                    },
                    {
                        'v': 3,
                        'c': [
                            {
                                'v': 4,
                                'c':[]
                            },
                            {
                                'v': 5,
                                'c': []
                            }
                        ]
                    }
                ]
            },
            {
                'v': 6,
                'c': [
                    {
                        'v': 7,
                        'c': []
                    },
                    {
                        'v': 8,
                        'c': []
                    }
                ]
            }
        ]
    }
    print(tree_json)
    tree = json_to_tree(tree_json)
    set_tree_indices(tree)
    print(tree.pretty_string())
    adj_matrix = tree_to_adj(tree.size(), tree, True, False)
    dist_mat = tree_to_dist_mat(tree, directed=False)
    print(dist_mat)
    pass

import numpy as np
import scipy

def generate_random_numbers(mean, sigma, n=30):
    margin = float(sigma) * 0.05
    rd1 = np.random.uniform(0-margin, margin)
    m2 = float(mean) * 0.03
    rd2 = np.random.uniform(0-m2, m2)
    samples = np.random.normal(mean+rd2, sigma+rd1, size=n)
    return samples