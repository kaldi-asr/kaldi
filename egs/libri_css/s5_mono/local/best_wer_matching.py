#!/usr/bin/env python3
# Copyright   2020   Desh Raj
# Apache 2.0.

import sys, io
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

# This class stores all information about a ref/hyp matching
class WerObject:
    # By default, we set the errors to very high values to
    # handle the error case.
    id = ''
    ref_id = ''
    hyp_id= ''
    wer = 0
    num_ins = 0
    num_del = 0
    num_sub = 0
    wc = 0

    def __init__(self, line):
        self.id, details = line.strip().split(maxsplit=1)
        tokens = details.split()
        self.wer = float(tokens[1])
        self.wc = int(tokens[5][:-1])
        self.num_ins = int(tokens[6])
        self.num_del = int(tokens[8])
        self.num_sub = int(tokens[10])
        self.ref_id, self.hyp_id = self.id[1:].split('h')


infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# First we read all lines and create a list of WER objects
wer_objects=[]
for line in infile:
    if not line or line.isspace():
        continue
    wer_object = WerObject(line)
    wer_objects.append(wer_object)

# Now we create a matrix of costs (WER) which we will use to solve
# a linear sum assignment problem
sort(wer_objects, key=lambda x: x.ref_id)
wer_object_matrix = [list(g) for ref_id, g in itertools.groupby(wer_objects, lambda x: x.ref_id)]
if len(wer_object_matrix) > len(wer_object_matrix[0]):
    # More references than hypothesis; take transpose
    wer_object_matrix = [*zip(*wer_object_matrix)]
wer_matrix = np.array([[1000 if math.isnan(obj.wer) else obj.wer 
    for obj in row] 
    for row in wer_object_matrix])

# Solve the assignment problem and compute WER statistics
row_ind, col_ind = linear_sum_assignment(wer_matrix)
total_ins = 0
total_del = 0
total_sub = 0
total_wc = 0
for row,col in zip(row_ind,col_ind):
    total_ins += wer_object_matrix[row][col].num_ins
    total_del += wer_object_matrix[row][col].num_del
    total_sub += wer_object_matrix[row][col].num_sub
    total_wc += wer_object_matrix[row][col].wc
total_error = total_ins+total_del+total_sub
wer = float(100*total_error)/total_wc

# Write the final statistics to stdout
print ("%WER {:.2f} [ {} / {}, {} ins, {} del, {} sub ]".format(wer, total_error, total_wc,
    total_ins, total_del, total_sub))
