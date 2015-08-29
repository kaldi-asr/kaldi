#! /usr/bin/env python

from __future__ import print_function
import sys
import numpy

A = [];
for line in sys.stdin.readlines():
  line = line.strip();
  splits = line.split();
  x = float(splits[4]);
  A.append(x);

min_x = min(A);
max_x = max(A);
mean_x = sum(A) / len(A);
per10_x = numpy.percentile(A, 10);
per25_x = numpy.percentile(A, 25);
per50_x = numpy.percentile(A, 50);
per75_x = numpy.percentile(A, 75);

print("%5.2f %5.2f %5.2f" % (min_x, max_x, mean_x));
print("%5.2f %5.2f %5.2f %5.2f" % (per10_x, per25_x, per50_x, per75_x));
