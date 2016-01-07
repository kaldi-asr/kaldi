#!/usr/bin/python

# Copyright 2015 Hossein Hadian

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


import sys

if len(sys.argv) != 3:
  print ('This program finds the maximum phone duration (in frames) to be used'
        'in duration modeling.')
  print 'Usage: find-max-duration <alignment-file> <min-repeat-count>'
  sys.exit(1)

alifile = sys.argv[1]
min_repeat = int(sys.argv[2])
print 'Alignment file:', alifile
print 'Minimum required repeat count:', min_repeat

durations = {}  # Stores the count of each duration (which is integer)

# Count the frequency of each duration:
for line in open(alifile):
  for pair in line.rstrip().split(' ; '):
    split_pair = pair.split(' ');    
    if len(split_pair) == 3:
      split_pair = [split_pair[1], split_pair[2]]
    if len(split_pair) != 2:
      print "Bad line: ", line
      sys.exit(1)
    duration = int(split_pair[1]);
    if duration in durations:
      durations[duration] += 1
    else:
      durations[duration] = 1

# Find the maximum duration d for which there are at least min_repeat phones 
# for all durations d-1, d-2, ...
for duration in xrange(min(durations.keys()), max(durations.keys())):
  if (duration not in durations) or (durations[duration] < min_repeat) :
    print 'max-duration:', duration - 1
    sys.exit(0)

print 'max-duration:', max_duration
sys.exit(0)
