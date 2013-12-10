#!/usr/bin/python -u

# ./gen_hamm_dct.py
# script generateing NN initialization for training with TNet
#     
# author: Chao Weng 
#

import math, random
import sys




if (len(sys.argv) != 4):
  print >> sys.stderr, 'Usage....\n' 
  sys.exit(1) 

recur_layer = int(sys.argv[2])
nnet_in = open(sys.argv[1], 'r')
nnet_out = open(sys.argv[3], 'w')
cur_layer = 0 

for line in nnet_in:
  line_list = line.split()
  if (line_list[0] == '<sigmoid>'):
    cur_layer = cur_layer + 1
  if (line_list[0] == '<sigmoid>' and cur_layer == recur_layer):
    nnet_out.write('<recurrent> ' + str(line_list[1]) +  ' ' + str(line_list[2]) + '\n')
    nnet_out.write('[\n')
    #weight matrix
    for r in range(int(line_list[1])):
      for c in range(int(line_list[2])):
        nnet_out.write(str(1/math.sqrt(float(line_list[2]))*random.gauss(0.0, 1.0)))
        nnet_out.write(' ')
      nnet_out.write('\n') 
    nnet_out.write(']\n')
    #bias vector
    nnet_out.write('[ ')
    for c in range(int(line_list[2])):
      nnet_out.write(str(random.random()/5.0-4.1)) 
      nnet_out.write(' ') 
    nnet_out.write(']\n')
  else:
    nnet_out.write(line)
  
