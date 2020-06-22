#!/usr/bin/env python3
# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0
#
# When using chain model, we can use this script to handle the result and do normalization, after script we get result file

import argparse
import numpy as np
def read_scp(filename):
	f = open(filename,'r')
	filedict = {}
	for line in f.readlines():
		filedict[line.split()[0]] = (float)(line.split()[1])
	return filedict


def nol(x, axis = 0):
	x = np.array(x)
	row_max = x.max(axis = axis)
	row_min = x.min(axis = axis)
	s=(x - row_min) / (row_max - row_min)
	return s.tolist()

def main(arg):
	score_src_dict = read_scp(arg.score)
	score = dict(zip(score_src_dict.keys(),nol(score_src_dict.values())))
	result = read_scp(arg.result)
	for key in score.keys():
		if result[key]-0 < 1e-5:
			score[key] = 0
	for key in score.keys():
		print(key + " " + str(score[key]))
if __name__ == '__main__':
	parser =argparse.ArgumentParser(
        description = 'When using chain model, we can use this script to handle the result and do normalization, after script we get result file')
	parser.add_argument('score', help = 'score.txt')
	parser.add_argument('result', help = 'result.txt')

	FLAGS = parser.parse_args()

	main(FLAGS)

