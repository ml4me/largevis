from rpforest import RPForest
import numpy as np
from heapq import heappush, heappop, heappushpop, nsmallest


np.random.seed(42)
def lvnn(fp, nt=3, k=5, iter= 10, leaves=50):

    nn=np.zeros((fp.shape[0],k))-1

    print(' start Tree build')
    model = RPForest(leaf_size=leaves, no_trees=nt)
    model.fit(fp)
    for i in range(0,fp.shape[0]):
        nn [i,:] =model.query(fp [i,],k)
    t=0
    while t<iter:
        t +=1
        print(t)
        old_nn=nn
        nn=nn*0-1
        for i in range(0,fp.shape[0]):
            h= set()
            for j in range(0,k):
                ji=old_nn[i,j]
                for l in range(0,k):
                    li=old_nn[ji,l]
                    d=-np.linalg.norm(fp [i,:]-fp [li,:])
                    h.update((d,li))
                nn [i,:]=np.array(map(lambda x: x [1] ,nsmallest(k,h)))
    nn


fp = np.random.randn(1000,10)


from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(2, 0.4)
neigh.fit(fp)
neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
