import numpy as np
import sicpy as sp

input_dim = 40
ivector_dim = 100
chain_splice_indices = [[-1,0,1], [-1,0,1,2], [-3,0,3], [-3,0,3], [-6,-3,0], [0]]
chain_widths = np.array([len(indices) for indices in chain_splice_indices])
chain_relu_dim = 576
xent_splice_indices = [[-2,-1,0,1,2], [-1,2], [-3,3], [-7,2], [0]]
xent_widths = np.array([len(indices) for indices in xent_splice_incdices])

def param_difference(new_relu_dim):
    """
    Returns difference in number of parameters between for new_relu_dim
    """
    relu_dim * (
        xent_widths[0]*(input_dim) + ivector_dim + xent_width[-1]*(output_dim)
        + relu_dim * sum(chain_widths[])
        )
