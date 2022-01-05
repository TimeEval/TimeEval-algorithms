# Code adapted from https://github.com/ptrus/suffix-trees
# Using notes from https://www.cs.helsinki.fi/u/tpkarkka/opetus/10s/spa/lectures.pdf
# McCreight, Edward M. "A space-economical suffix tree construction algorithm." - ACM, 1976.
# Tarzan implementation from https://www.cs.ucr.edu/~eamonn/sigkdd_tarzan.pdf

import numpy as np
from math import floor, ceil
import warnings
from typing import Optional

from .original_discretization import original_discretization


def TARZAN(r, x, l, a, relative=False):
    """
    Modified Implementation of TARZAN, see https://www.cs.ucr.edu/~eamonn/sigkdd_tarzan.pdf
    """
    print("Discretizing")
    R, X = DISCRETIZE_TIME_SERIES(r, x, l, a)

    print("Creating suffix tree for train and test and annotating them")
    # Compute suffix trees for both R and X
    Rtree = SuffixTree(R)
    Xtree = SuffixTree(X)

    print("Searching anomalies")
    # suggestion from paper: l2 < log_a(|X|)
    l2 = np.floor(np.log(len(X)) / np.log(a)).astype(int)
    print(f"Using window size l2={l2} for sliding over discretized TS")
    score = [0]*floor((l2-1)/2)
    for i in range(len(X)-l2+1):
        w = X[i: i+l2]

        # find word in X
        node = Xtree.find(w, list_all=False, index=False)

        # compute expectation in R
        E = compute_expectation(w, R, X, Rtree)
        # compute score
        if relative:
            z = (node.frequency - E)/max(abs(node.frequency), abs(E))
        else:
            z = node.frequency - E

        score.append(abs(z))
    score += [0]*ceil((l2-1)/2)

    return score

def DISCRETIZE_TIME_SERIES(ts_reference, ts_new, l: Optional[int] = None, a: Optional[int] = None):
    """
    Converts time series to symbolic representation using sybolic constructor
    """
    r = original_discretization(ts_reference, l, a)
    x = original_discretization(ts_new, l, a)

    return r, x

def compute_expectation(w, R, X, Rtree):
    """
    Compute expectation of word in string.
    """
    # compute scale factor
    alpha = lambda m: (len(X) - m + 1)/(len(R) - m + 1)

    m = len(w)

    # if w in Rtree then count occurences
    node = Rtree.find(w, list_all=False, index=False)
    if isinstance(node, _Node):
        E = alpha(m)*node.frequency
    # otherwise look for largest interval such that all string in that interval are in Rtree
    else:
        l_exist = False
        for l in range(m-2, 1, -1): # 1 < l < m-1
            p = _lower_mod_prod(Rtree, w, 0, m-l, l)
            if p > 0:
                l_exist = True
                num = p
                break

        if l_exist: # if interval exists
            den = _lower_mod_prod(Rtree, w, 1, m-l, l-1)
            # E = alpha(m) * ((num+1e-10)/(den+1e-10)) # may divide by zero?
            try:
                E = alpha(m) * (num / den)
            except ZeroDivisionError:
                import sys
                print(f"Zero division error, maximizing expectation E={sys.maxint} for word w={w}")
                E = sys.maxint
        else: # otherwise probability of symbols
            E = (len(X) - m + 1)* _symbol_prob(Rtree, R, w)
    return E

def _lower_mod_prod(tree, w, j_min, j_max, l):
    """
    Utility function used to compute \prod_{j=jmin}^{jmax} f_tree(w[j:j+l])
    """
    prod = 1
    for j in range(j_min, j_max+1):
        node2 = tree.find(w[j:j+l], list_all=False, index=False)
        if node2==False:
            return 0
        else:
            prod *= node2.frequency
    return prod

def _symbol_prob(Rtree, R, w):
    """
    Utility function used to compute \prod_{j=1}^{m} w_i
    """
    prod = 1
    for symbol in w:
        node2 = Rtree.find(symbol, list_all=False, index=False)
        if node2==False:
            return 0
        else:
            prod *= node2.frequency/len(R)
    return prod

class _Node():
    """
    Class representing a Node in the Suffix tree.
    """
    def __init__(self, idx=-1, parentNode=None, depth=-1):
        # Links
        self.child = []            # list of children nodes
        self._slink = None         # used for McCreight method
        # Properties
        self.idx = idx             # idx: node index
        self.depth = depth         # depth: length of string represented by node
        self.parent = parentNode   # parent: parent node of this node
        self.frequency = 0         # frequency: used for tarzan algorithm
        self.z = None              # z-metric: used for tarzan algorithm

    def _get_child(self, suffix):
        """
        Returns node if suffix is edge to child and False otherwise
        """
        for node,_suffix in self.child:
            if suffix == _suffix:
                return node
        return False

    def _add_child(self, node, suffix):
        """
        Add child to node with edge suffix
        """
        tl = self._get_child(suffix)
        if tl:
            self.child.remove((tl,suffix))
        self.child.append((node,suffix))

    def is_leaf(self):
        """
        Check if node is leaf by looking for children
        """
        return self.child == []

    def _get_leaves(self):
        """
        Find leaves from node by working down tree.
        """
        if self.is_leaf():
            return [self]
        else:
            return [x for (n,_) in self.child for x in n._get_leaves()]

    def _get_slink(self):
        """
        Get suffix link node.
        """
        if self._slink != None:
            return self._slink
        else:
            return False


class SuffixTree():
    """
    Class representing the suffix tree.

    Note that each edge only stores one letter, even though edge may represent a word.
    """
    def __init__(self, string, method='McCreight'):

        self._create_root_node()

        # Add terminating symbol (required by McCreight algorithm)
        string += '$'
        self.word = string

        # Build suffix tree
        if method == 'McCreight':
            self._build_McCreight(string)
        elif method == 'Brute':
            self._build_BruteForce(string)
        else:
            warnings.warn('Unrecognised method, using McCreight')
            self._build_McCreight


    def _create_root_node(self):
        """
        Construct root node
        """
        self.root = _Node()
        self.root.depth = 0
        self.root.idx = 0
        self.root.parent = self.root
        self.root._slink = self.root # see Lemma 4.4 in https://www.cs.helsinki.fi/u/tpkarkka/opetus/10s/spa/lectures.pdf

    def _create_node(self, T, u, d):
        """
        Add node to tree. Think of this as adding node v between u and
        u.parent
        """
        i = u.idx
        p = u.parent
        v = _Node(idx=i, depth=d)
        v._add_child(u, T[i+d]) # add child
        u.parent = v
        p._add_child(v, T[i+p.depth]) # add node to child of parent
        v.parent = p
        return v

    def _create_leaf(self, T, i, u, d):
        """
        Add leaf to tree. Add leaf below u.
        """
        w = _Node()
        w.idx = i
        w.depth = len(T) - i
        u._add_child(w, T[i + d])
        w.parent = u

        # update annotations
        w.frequency = 1
        def annotate(node):
            node.frequency += 1
            if node.idx != 0:
                annotate(node.parent)
        annotate(w.parent)
        return w

    def _compute_slink(self, T, u):
        """
        Compute suffix link of a particular node. See slide 142 of
        https://www.cs.helsinki.fi/u/tpkarkka/opetus/10s/spa/lectures.pdf
        """
        v = u.parent._get_slink()
        while v.depth < u.depth - 1:
            v = v._get_child(T[u.idx + v.depth + 1])
        if v.depth > u.depth - 1:
            v = self._create_node(T, v, u.depth-1)
        u._slink = v

    def _build_BruteForce(self, T):
        """
        Builds a Suffix tree using Brute Force O(n^2) algorithm. See slide 138 of
        https://www.cs.helsinki.fi/u/tpkarkka/opetus/10s/spa/lectures.pdf
        """
        u, d = self.root, 0
        for i in range(len(T)): # Insert suffix T[i,...,n]
            # Check if child has first letter of suffix
            while u.depth == d and isinstance(u._get_child(T[d+i]), _Node):
                # Move to child node
                u, d = u._get_child(T[d+i]), d + 1
                # Edge may represent word even though child only stores first letter.
                # Traverse through word represented by edge.
                while d < u.depth and T[u.idx + d] == T[i + d]:
                    d = d + 1
            if d < u.depth: # We are in the middle of an edge
                # Need to split current edge and add node in middle. Make existing child
                # the child of new node.
                u = self._create_node(T, u, d)

            # Add new leaf represents suffix T[i,...,n]
            self._create_leaf(T, i, u, d)
            u, d = self.root, 0

    def _build_McCreight(self, T):
        """
        Builds a Suffix tree using McCreight O(n) algorithm. See slide 145 of
        https://www.cs.helsinki.fi/u/tpkarkka/opetus/10s/spa/lectures.pdf
        """
        u, d = self.root, 0
        for i in range(len(T)): #insert suffix T[i,...,n]
            # Check if child has first letter of suffix
            while u.depth == d and isinstance(u._get_child(T[d+i]), _Node):
                # Move to child node
                u, d = u._get_child(T[d+i]), d + 1
                # Edge may represent word even though child only stores first letter.
                # Traverse through word represented by edge.
                while d < u.depth and T[u.idx + d] == T[i + d]:
                    d = d + 1
            if d < u.depth: # We are in the middle of an edge
                # Need to split current edge and add node in middle. Make existing child
                # the child of new node.
                u = self._create_node(T, u, d)
                # Note when creating node, slink initialised as None

            # Add new leaf represents suffix T[i,...,n]
            self._create_leaf(T, i, u, d)

            # Compute suffix link of node u
            if not u._get_slink():
                self._compute_slink(T, u)

            u, d = u._get_slink(), d - 1 if d > 0 else 0

    def _edgeLabel(self, node, parent):
        """
        Each node only stores first letter of edge. This function returns full edge label.
        """
        return self.word[node.idx + parent.depth : node.idx + node.depth]

    def _nodeLabel(self, node):
        """
        Each node only stores first letter of edge. This function returns concatenation of all edges from
        root
        """
        return self.word[node.idx : node.idx + node.depth]

    def _get_word_start_index(self, idx):
        """
        Returns the index of the string based on node's starting index
        """
        i = 0
        for _idx in self.word_starts[1:]:
            if idx < _idx:
                return i
            else:
                i+=1
        return i

    def find(self, y, list_all=True, index=True):
        """
        Finds substring y in T.
        If list_all=True and index=True then returns starting positions and [] is not in tree.
        If list_all=True and index=False then returns leaves nodes.
        If list_all=False and index=True then returns bool.
        If list_all=False and index=False then node.
        """
        # start at root
        node = self.root

        while True:
            # take edge between node and parent
            edge = self._edgeLabel(node, node.parent)
            # if edge is our search query then we have done, return all leaves
            if edge.startswith(y):
                break

            # chop off start of y as we find matches
            i = 0
            while(i < len(edge) and edge[i] == y[0]):
                y = y[1:]
                i += 1

            if i != 0:
                # we have found an edge which is prefix of our query
                if i == len(edge) and y != '':
                    # find child and repeat
                    node = node._get_child(y[0])
                    if not node:
                        if list_all:
                            return []
                        else:
                            return False
                    else:
                        pass
                else:
                    if list_all:
                        return []
                    else:
                        return False
            else:
                # then i </ len(edge), maybe child is leaf
                node = node._get_child(y[0])
                if not node:
                    if list_all:
                        return []
                    else:
                        return False

        if list_all:
            # once found substring in tree, all leaves are occurences of substring.
            leaves = node._get_leaves()
            if index:
                return [n.idx for n in leaves]
            else:
                return [n for n in leaves]
        else:
            if index:
                return True
            else:
                return node
