import numpy as np
import sys
from scipy.stats import entropy

def mtd_delta_t(postgram, delta_t=10):
    '''
    Basically computes the following formulat
MTD(\Delta t) = \frac{1}{T-\Delta t} \sum_{t=\Delta t}^{T} D(p_{t-\Delta t}, p_t)    
    '''
    
    mtd_delta_t=0.0
    T=postgram.shape[0]
    for i in xrange(delta_t, T):

        pk=postgram[i-delta_t,:]
        qk=postgram[i,:]

        # symmetric KLDIV
        mtd_delta_t=mtd_delta_t+(entropy(pk,qk)+entropy(qk,pk))

    return mtd_delta_t/(T-delta_t)

def compute_mtd(postgram, delta_t=range(10,81,5)):
    
    # for short utterances
    T=postgram.shape[0]
    new_delta_t=filter(lambda x: x<T, delta_t)
    delta_t=new_delta_t

    mtd_vect=np.zeros((len(delta_t),1))
    i=0
    for dt in delta_t:
        mtd_vect[i,:]=mtd_delta_t(postgram, dt)
        i=i+1

    return mtd_vect

