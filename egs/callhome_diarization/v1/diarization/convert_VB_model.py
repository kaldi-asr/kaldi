#!/usr/bin/env python

# Copyright 2019  Zili Huang
# Apache 2.0

# This script is called by diarization/convert_VB_model.sh.
# It converts diagonal UBM and ivector extractor to numpy 
# array format

import numpy as np
import pickle
import sys

def load_dubm(dubm_text):
    para_dict = {}
    with open(dubm_text, 'r') as fh:
        content = fh.readlines()
    state = 0
    data_array = []

    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        if state == 0:
            if len(line_split) == 1:
                continue
            elif len(line_split) == 2 and line_split[1] == "[": # Start of a multi-line matrix like <MEANS_INVVARS> and <INV_VARS> 
                para_name = line_split[0]
                state = 1
                data_array = []
            elif len(line_split) >= 3 and line_split[1] == "[" and line_split[-1] == "]": # Single line vector like <WEIGHTS>
                para_name = line_split[0]
                data_list = []
                for i in range(2, len(line_split) - 1):
                    data_list.append(float(line_split[i]))
                data_list = np.array(data_list)
                para_dict[para_name] = data_list
            else:
                raise ValueError("Condition not defined.")
        elif state == 1:
            if line_split[-1] == "]": # End of a multi-line matrix like <MEANS_INVVARS> and <INV_VARS>
                data_list = []
                for i in range(len(line_split) - 1):
                    data_list.append(float(line_split[i]))
                data_list = np.array(data_list)
                data_array.append(data_list)
                data_array = np.array(data_array)
                para_dict[para_name] = data_array
                state = 0
            else:
                data_list = []
                for i in range(len(line_split)):
                    data_list.append(float(line_split[i]))
                data_list = np.array(data_list)
                data_array.append(data_list)
        else:
            raise ValueError("Condition not defined.")
    return para_dict 

def load_ivector_extractor(ie_text):
    para_dict = {}
    with open(ie_text, 'r') as fh:
        content = fh.readlines()
    state = 0
    data_3dmatrix = []

    for line in content:
        line = line.strip('\n')
        if line == "<SigmaInv> [":
            break
        if state == 0:
            if not line.startswith("<M>"):
                continue
            else:
                state = 1
                data_matrix = []
        elif state == 1:
            line_split = line.split()
            if line_split[0] == "[":
                data_matrix = []
                continue
            elif line_split[-1] == "]":
                data_array = []
                for i in range(len(line_split)-1):
                    data_array.append(float(line_split[i]))
                data_matrix.append(data_array)
                data_3dmatrix.append(data_matrix)
            else:
                data_array = []
                for i in range(len(line_split)):
                    data_array.append(float(line_split[i]))
                data_matrix.append(data_array)
        else:
            raise ValueError("Condition not defined.")
    para_dict['M'] = np.array(data_3dmatrix)
    return para_dict 

def save_dict(para_dict, output_filename):
    with open(output_filename, 'wb') as fh:
        pickle.dump(para_dict, fh)
    return 0

def main():
    dubm_model = sys.argv[1]
    ivec_extractor_model = sys.argv[2]
    output_dir = sys.argv[3]

    # the diagonal ubm parameter includes <GCONSTS>, <WEIGHTS>, <MEANS_INVVARS>, <INV_VARS>
    dubm_para = load_dubm(dubm_model)
    save_dict(dubm_para, "{}/diag_ubm.pkl".format(output_dir))

    # the ivector extractor parameter is a 3d matrix of shape [num-gaussian, feat-dim, ivec-dim]
    ie_para = load_ivector_extractor(ivec_extractor_model)
    save_dict(ie_para, "{}/ie.pkl".format(output_dir))
    return 0

if __name__ == "__main__":
    main()
