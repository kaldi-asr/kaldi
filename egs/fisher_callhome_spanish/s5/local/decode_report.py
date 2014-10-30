#!/usr/bin/env python

# Author : Gaurav Kumar (Johns Hopkins University)
# Gets a report on what the best word error rate was and which iteration
# led to it. This is needed both for reporting purposes and for setting
# the acoustic scale weight which extracting lattices. 
# This script is specific to my partitions and needs to be made more general
# or modified

import subprocess
import os

decode_directories = ['exp/tri5a/decode_dev',
                        'exp/tri5a/decode_test',
                        'exp/tri5a/decode_dev2',
                        'exp/sgmm2x_6a/decode_dev_fmllr',
                        'exp/sgmm2x_6a/decode_test_fmllr',
                        'exp/sgmm2x_6a/decode_dev2_fmllr',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_it4',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_it4',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_it4',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_fmllr_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_fmllr_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_fmllr_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev_fmllr_it4',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_fmllr_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_fmllr_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_fmllr_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_dev2_fmllr_it4',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it1',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it2',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it3',
                        'exp/sgmm2x_6a_mmi_b0.2/decode_test_fmllr_it4'
                        ]

def get_best_wer(decode_dir):
    best_iteration = 0
    best_wer = 100.0
    for i in range(16):
        if os.path.isfile(decode_dir + "/wer_" + str(i)):
            result = subprocess.check_output("tail -n 3 " + decode_dir + "/wer_" + str(i), shell=True)
            wer_string = result.split("\n")[0]
            wer_details = wer_string.split(' ')
            # Get max WER
            wer = float(wer_details[1])
            if wer < best_wer:
                best_wer = wer
                best_iteration = i
    return best_iteration, best_wer

for decode_dir in decode_directories[:6]:
    print decode_dir
    print get_best_wer(decode_dir)

# Separate processing for bMMI stuff
best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[6:10]:
    iteration, wer = get_best_wer(decode_dir)
    if wer < best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)

best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[10:14]:
    iteration, wer = get_best_wer(decode_dir)
    if wer < best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)

best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[14:18]:
    iteration, wer = get_best_wer(decode_dir)
    if wer < best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)

best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[18:22]:
    iteration, wer = get_best_wer(decode_dir)
    if wer <= best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)

best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[22:26]:
    iteration, wer = get_best_wer(decode_dir)
    if wer <= best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)

best_wer = 100.0
best_dir = ""
best_iteration = 0

for decode_dir in decode_directories[26:]:
    iteration, wer = get_best_wer(decode_dir)
    if wer <= best_wer:
        best_wer = wer
        best_dir = decode_dir
        best_iteration = iteration

print best_dir
print (best_iteration, best_wer)
