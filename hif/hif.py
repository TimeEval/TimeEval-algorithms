"isolated forest functions"
__author__ = 'P-F.M., February 217, derived from Matias Carrasco Kind code (https://github.com/mgckind/iso_forest)'
import numpy as np
import random as rn
from collections import Counter
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool

def EuclideanDist(x,y):
    return np.sqrt(np.sum((x - y) ** 2))

def dist2set(x, X):
    l=len(X)
    ldist=[]
    for i in range(l):
        ldist.append(EuclideanDist(x,X[i]))
    return ldist

def c_factor(n) :
    if(n<2):
        n=2
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

class hiForest(object):

    def buildTree(self, i):
        ix = rn.sample(range(self.nobjs), self.sample)
        X_p = self.X[ix]
        return(hiTree(X_p, 0, self.limit))

    def __init__(self,X, ntrees,  sample_size, limit=None, nCore=1):
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.nCore=nCore
        self.X_in=None

        if limit is None:
            self.limit = int(np.ceil(1.2*np.log2(self.sample)))
        self.c = c_factor(self.sample)
        self.Trees = []
        for i in range(self.ntrees):
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(hiTree(X_p, 0, self.limit))

    def computeScore_paths(self, X_in = None):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i],self.Trees[j]).path*1.0
            Eh = h_temp/self.ntrees
            S[i] = 2.0**(-Eh/self.c)
        return S

    def computeScore(self, i):
        h_temp = 0
        for j in range(self.ntrees):
            h_temp += PathFactor(self.X_in[i], self.Trees[j]).path * 1.0
        Eh = h_temp / self.ntrees
        return 2.0 ** (-Eh / self.c)


    def computeScore_pathsPool(self, X_in = None):
        pool = Pool(self.nCore)
        if X_in is None:
            X_in = self.X
        self.X_in=X_in
        print(np.shape(X_in), self.nCore)
        L=len(X_in)
        tab = list(range(L))
        print(tab)
        S=pool.map(self.computeScore, tab)
        return S

    def computeScore_paths_single(self, x):
        S = np.zeros(self.ntrees)
        for j in range(self.ntrees):
            path =  PathFactor(x,self.Trees[j]).path*1.0
            S[j] = 2.0**(-1.0*path/self.c)
        return S

    def computeScore_paths_single_with_labs(self, x):
        S = np.zeros(self.ntrees)
        labs=[]
        for j in range(self.ntrees):
            pf=PathFactor(x,self.Trees[j])
            path =  pf.path*1.0
            S[j] = 2.0**(-1.0*path/self.c)
            labs.append(pf.labs)
        return S, labs


    def computeAggScore(self, x):
        S = np.zeros(self.ntrees)
        labsCount = Counter([])
        ldist=[]
        ldist_a=[]
        for j in range(self.ntrees):
            pf=PathFactor(x,self.Trees[j])
            path =  pf.path*1.0
            S[j] = 2.0**(-1.0*path/self.c)
            labsCount=labsCount+pf.labs
            if(len(pf.ldist)>0):
                ldist.append(np.mean(pf.ldist))
            if(len(pf.ldist_a)>0):
                ldist_a.append(np.mean(pf.ldist_a, axis=0))
        meanDist=0
        if(len(ldist)>0):
            meanDist=np.mean(ldist)
        meanDist_r = 0
        if(len(ldist_a)>0):
            meanDist_a = np.mean(ldist_a, axis=0)
            if(meanDist_a>0):
                meanDist_r=meanDist/(meanDist_a)
                #meanDist_r = 1.0 / (meanDist_a)
        return np.mean(S), labsCount, meanDist, meanDist_r


    def addAnomaly(self, x, lab):
        for j in range(self.ntrees):
            pf=PathFactor(x,self.Trees[j])
            pf.addAnomaly(x, lab, self.Trees[j].root)

    def computeAnomalyCentroid(self):
        for j in range(self.ntrees):
            self.Trees[j].root.computeAnomalyCentroid()

    def getAverageBucketSize(self):
        szb=0
        nbb=0
        for j in range(self.ntrees):
            s,n=self.Trees[j].root.getAverageBucketSize()
            szb+=s
            nbb+=n
        out=0
        if nbb>0:
            out=szb/nbb
        return out, nbb/self.ntrees


class Node(object):
    def __init__(self, X, q, p, e, left, right, node_type = '' ):
        self.e = e
        self.size = len(X)
        self.X = X # to be removed
        self.q = q
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type
        self.C = None
        self.Ca = None
        self.labs=[]
        self.Xanomaly=[]
        if(node_type == 'exNode' and self.size>0):
            self.C = np.mean(X, axis=0)


    def computeAnomalyCentroid(self):
        if self.ntype == 'exNode':
            if(len(self.Xanomaly)>0):
                self.Ca = np.mean(self.Xanomaly, axis=0)
        else:
            self.left.computeAnomalyCentroid()
            self.right.computeAnomalyCentroid()

    def getAverageBucketSize(self):
        if self.ntype == 'exNode':
            return self.size, 1
        else:
            s1, n1 = self.left.getAverageBucketSize()
            s2, n2 = self.right.getAverageBucketSize()
            return s1+s2, n1+n2

class hiTree(object):

    """
    Unique entries for X
    """

    def __init__(self,X,e,l):
        self.e = e # depth
        self.X = X #save data for now
        self.size = len(X) #  n objects
        self.Q = np.arange(np.shape(X)[1], dtype='int') # n dimensions
        self.l = l # depth limit
        self.p = None
        self.q = None
        self.exnodes = 0
        self.labs=[]
        self.root = self.make_tree(X,e,l)
        

    def make_tree(self,X,e,l):
        self.e = e
        if e >= l or len(X) <= 1:
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.q, self.p, e, left, right, node_type = 'exNode' )
        else:
            self.q = rn.choice(self.Q)
            self.p = rn.uniform(X[:,self.q].min(),X[:,self.q].max())
            w = np.where(X[:,self.q] < self.p,True,False)
            return Node(X, self.q, self.p, e,\
            left=self.make_tree(X[w],e+1,l),\
            right=self.make_tree(X[~w],e+1,l),\
            node_type = 'inNode' )

    def get_node(self, path):
        node = self.root
        for p in path:
            if p == 'L' : node = node.left
            if p == 'R' : node = node.right
        return node


class PathFactor(object):
    def __init__(self,x,hitree):
        self.path_list=[]
        self.labs = []
        self.ldist = []
        self.ldist_a = []
        self.x = x
        self.e = 0
        self.path = self.find_path(hitree.root)


    def find_path(self,T):
        if T.ntype == 'exNode':
            self.labs = Counter(T.labs)
            if not (T.C is None):
                self.ldist.append(EuclideanDist(self.x, T.C))
            if not (T.Ca is None):
                self.ldist_a.append(EuclideanDist(self.x, T.Ca))
            sz=T.size
            if(sz==0):
                sz+=1
            for key in self.labs:
                self.labs[key] /= sz
            if T.size == 1:
                return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            a = T.q
            self.e += 1
            if self.x[a] < T.p:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)

    def addAnomaly(self, x, lab, T):
        if T.ntype == 'exNode':
            T.labs.append(lab)
            T.Xanomaly.append(x)
        else:
            a = T.q
            if self.x[a] < T.p:
                return self.addAnomaly(x, lab, T.left)
            else:
                return self.addAnomaly(x, lab, T.right)


def all_branches(node, current=[], branches = None):
    current = current[:node.e]
    if branches is None: branches = []
    if node.ntype == 'inNode':
        current.append('L')
        all_branches(node.left, current=current, branches=branches)
        current = current[:-1]
        current.append('R')
        all_branches(node.right, current=current, branches=branches)
    else:
        branches.append(current)
    return branches


def branch2num(branch, init_root=0):
    num = [init_root]
    for b in branch:
        if b == 'L':
            num.append(num[-1] * 2 + 1)
        if b == 'R':
            num.append(num[-1] * 2 + 2)
    return num
