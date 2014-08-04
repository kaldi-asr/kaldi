#!/usr/bin/python
#
# Copyright  2014 Nickolay V. Shmyrev 
# Apache 2.0


import sys

words = set()
for line in open(sys.argv[1]):
    items = line.split()
    words.add(items[0])

for line in sys.stdin:
    items = line.split()
    new_items = []
    i = 1
    while i < len(items):
	if i < len(items) - 1 and items[i+1][0] == '\'' and items[i] + items[i+1] in words:
	    new_items.append(items[i] + items[i+1])
	    i = i + 1
	else:
	    new_items.append(items[i])
	i = i + 1
	
    print items[0], " ".join(new_items)
    
