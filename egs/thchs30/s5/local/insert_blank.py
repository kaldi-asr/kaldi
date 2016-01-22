#!/usr/bin/python2.6

import sys

for l in sys.stdin:
    l=l.strip()
    ll=l.split()
    lk=ll[0]
    for v in ll[1:]:
        v = v.decode('utf-8')
        for i in v:
           lk= lk + ' ' + i
        
    print lk.encode('utf-8')
