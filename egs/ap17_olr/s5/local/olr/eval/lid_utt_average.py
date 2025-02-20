#!/usr/bin/env python

import sys, os, os.path

data = sys.argv[1]
data_base = sys.argv[2]

len_dict = {}
with open(data + '/feats.len', 'r') as utt_lens:
	for utt_len in [line.strip().split() for line in utt_lens]:
                len_dict[utt_len[0]] = utt_len[1]

dir = 'lid_net_output'
for parent,dirnames,filenames in os.walk(dir):
	for file in filenames:
		if file.endswith('ark.utt') and data_base in file:
			new_file = file + '_average'
			w_f = open(dir+'/'+new_file,'w')
			with open(dir+'/'+file, 'r') as lines:
				for col in [line.strip().split() for line in lines]:
					w_f.write(col[0] + ' [ ')
					utt_id = col[0]
					for i in range(2, len(col)-1):
						w_f.write(str(float(col[i])/float(len_dict[utt_id])) + ' ')
					w_f.write(']\n')
			w_f.close()
			print "Utter level results for " + data_base + " done."
