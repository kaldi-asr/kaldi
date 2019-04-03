#!/usr/bin/env python

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
            elif len(line_split) == 2 and line_split[1] == "[":
                para_name = line_split[0]
                state = 1
                data_array = []
            elif len(line_split) >= 3 and line_split[1] == "[" and line_split[-1] == "]": # One line vector
                para_name = line_split[0]
                data_list = []
                for i in range(2, len(line_split) - 1):
                    data_list.append(float(line_split[i]))
                data_list = np.array(data_list)
                para_dict[para_name] = data_list
            else:
                raise ValueError("Condition not defined.")
        elif state == 1:
            if line_split[-1] == "]":
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
    data_matrix = []
    data_array = []

    for line in content:
        line = line.strip('\n')
        if line == "<SigmaInv> [":
            break
        if state == 0:
            if line != "<M> 1024  [":
                continue
            else:
                state = 1
        elif state == 1:
            line_split = line.split()
            if line_split[0] == "[":
                continue
            elif line_split[-1] == "]":
                data_array = []
                for i in range(len(line_split)-1):
                    data_array.append(float(line_split[i]))
                data_matrix.append(data_array)
                data_3dmatrix.append(data_matrix)
                data_matrix = []
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

def judge_case(txt_model):
    with open(txt_model, 'r') as fh:
        first_line = fh.readline()
    model_type = first_line.split()[0]
    if model_type == "<DiagGMM>":
        return 1
    elif model_type == "<IvectorExtractor>":
        return 2
    else:
        return 0

def main():
    # The txt version of diagonal UBM and i-vector extractor. See gmm-global-copy 
    # and ivector-extractor-copy for details. (ivector-extractor-copy is not
    # supported in the official kaldi, so you have to use my kaldi)
    txt_model = sys.argv[1]
    output_dir = sys.argv[2]
    model_type = judge_case(txt_model)

    if model_type == 1: # DiagGMM
        dubm_para = load_dubm(txt_model)
        save_dict(dubm_para, "{}/diag_ubm.pkl".format(output_dir))
    elif model_type == 2: # IvectorExtractor
        ie_para = load_ivector_extractor(txt_model)
        save_dict(ie_para, "{}/ie.pkl".format(output_dir))
    else:
        raise ValueError("Condition not defined.")
    return 0

if __name__ == "__main__":
    main()
