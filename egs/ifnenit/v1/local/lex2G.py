import sys
import math

if len(sys.argv) == 2:
  inputFile = sys.argv[1]
else:
  sys.exit(1)

numWords = sum(1 for line in open(inputFile))
edgeP = -math.log(0.5 / numWords)
finP = -math.log(0.5)
constP = 95.0

print "0 1 #0 <eps>"
print "1 2 SIL SIL", edgeP
state=3
with open (inputFile) as f:
  for line in f:
    print 2, state, line.strip(), line.strip(), edgeP
    state = state + 1
for i in xrange(3, 3 + numWords):
  print i, state, "SIL SIL", edgeP
print state, state + 1, "#0 <eps>"
print state + 1, finP
