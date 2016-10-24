#!/usr/bin/python
import numpy as np

from scipy.stats import norm
from scipy.stats import entropy
from scipy.stats import kl_div


#element-wise KL smoothing


def kl_div_smooth(pk,qk): 
    pk_max_phone_prob = np.amax(pk)
    pk_max_phone_id = np.where (pk == pk_max_phone_prob)
    qk_weight_prob = qk[pk_max_phone_id]
    
    numerator=qk_weight_prob * entropy(pk,qk)
    denominator=qk_weight_prob
    return (numerator,denominator)


    
