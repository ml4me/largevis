from rpforest import RPForest
import numpy as np
from heapq import heappush, heappop, heappushpop, nsmallest


np.random.seed(42)
def lvnn(fp, nt=3, k=5, iter= 5, leaves=50):
    
    nn=np.zeros((fp.shape[0],k,2))-1
    
    print(' start Tree build')
    model = RPForest(leaf_size=leaves, no_trees=nt)
    model.fit(fp)
    for i in range(0,fp.shape[0]):
        nn[i,:,0] =model.query(fp [i,],k)
    
    t=0
    while t<iter:
        t +=1
        old_nn=nn
        for i in range(0,fp.shape[0]):
            h= set()
            for j in range(0,k):
                ji=old_nn[i,j,0]
                for l in range(0,k):
                    li=old_nn[ji,l,0]
                    d=-np.linalg.norm(fp [i,:]-fp [li,:])
                    h.update([(li,d)])
                nn [i,:,:]=np.array(nsmallest(k,h))
                
    csr=np.zeros((fp.shape[0]*k, 3))
    l=0
    for i in range(fp.shape[0]):
        for j in range(k):
            csr [l,0]=i
            csr [l,1]=nn[i,j, 0]
            csr [l,2]=nn[i,j, 1]
            l=l+ 1
    return csr

def testit():
    fp = np.random.randn(1000,1000)
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import kneighbors_graph
    import time
    print(' start skLearn')
    ms = time.time()*1000.0
    (kneighbors_graph(fp, 2))
    print(' time',(time.time()*1000.0)-ms)
    print(' start rp nn')
    ms = time.time()*1000.0
    (lvnn(fp, k=2))
    print(' time',(time.time()*1000.0)-ms)
    
