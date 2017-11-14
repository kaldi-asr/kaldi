#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2012 Mirko Hannemann BUT, mirko.hannemann@gmail.com

import sys
import codecs # for UTF-8/unicode

if len(sys.argv) != 2:
    print 'usage: reverse_arpa arpa.in'
    sys.exit()
arpaname = sys.argv[1]

#\data\
#ngram 1=4
#ngram 2=2
#ngram 3=2
#
#\1-grams:
#-5.234679	a -3.3
#-3.456783	b
#0.0000000	<s> -2.5
#-4.333333	</s>
#
#\2-grams:
#-1.45678	a b -3.23
#-1.30490	<s> a -4.2
#
#\3-grams:
#-0.34958	<s> a b
#-0.23940	a b </s>
#\end\

# read language model in ARPA format
try:
  file = codecs.open(arpaname, "r", "utf-8")
except IOError:
  print 'file not found: ' + arpaname
  sys.exit()

text=file.readline()
while (text and text[:6] != "\\data\\"): text=file.readline()
if not text:
  print "invalid ARPA file"
  sys.exit()
#print text,
while (text and text[:5] != "ngram"): text=file.readline()

# get ngram counts
cngrams=[]
n=0
while (text and text[:5] == "ngram"):
  ind = text.split("=")
  counts = int(ind[1].strip())
  r = ind[0].split()
  read_n = int(r[1].strip())
  if read_n != n+1:
    print "invalid ARPA file:", text
    sys.exit()
  n = read_n
  cngrams.append(counts)
  #print text,
  text=file.readline()

# read all n-grams order by order
sentprob = 0.0 # sentence begin unigram
ngrams=[]
inf=float("inf")
for n in range(1,len(cngrams)+1): # unigrams, bigrams, trigrams
  while (text and "-grams:" not in text): text=file.readline()
  if n != int(text[1]):
    print "invalid ARPA file:", text
    sys.exit()
  #print text,cngrams[n-1]
  this_ngrams={} # stores all read ngrams
  for ng in range(cngrams[n-1]):
    while (text and len(text.split())<2):
      text=file.readline()
      if (not text) or ((len(text.split())==1) and (("-grams:" in text) or (text[:5] == "\\end\\"))): break
    if (not text) or ((len(text.split())==1) and (("-grams:" in text) or (text[:5] == "\\end\\"))):
      break # to deal with incorrect ARPA files
    entry = text.split()
    prob = float(entry[0])
    if len(entry)>n+1:
      back = float(entry[-1])
      words = entry[1:n+1]
    else:
      back = 0.0
      words = entry[1:]
    ngram = " ".join(words)
    if (n==1) and words[0]=="<s>":
      sentprob = prob
      prob = 0.0
    this_ngrams[ngram] = (prob,back)
    #print prob,ngram.encode("utf-8"),back

    for x in range(n-1,0,-1):
      # add all missing backoff ngrams for reversed lm
      l_ngram = " ".join(words[:x]) # shortened ngram
      r_ngram = " ".join(words[1:1+x]) # shortened ngram with offset one
      if l_ngram not in ngrams[x-1]: # create missing ngram
        ngrams[x-1][l_ngram] = (0.0,inf)
        #print ngram, "create 0.0", l_ngram, "inf"
      if r_ngram not in ngrams[x-1]: # create missing ngram
        ngrams[x-1][r_ngram] = (0.0,inf)
        #print ngram, "create 0.0", r_ngram, "inf",x,n,h_ngram

      # add all missing backoff ngrams for forward lm
      h_ngram = " ".join(words[n-x:]) # shortened history
      if h_ngram not in ngrams[x-1]: # create missing ngram
        ngrams[x-1][h_ngram] = (0.0,inf)
        #print "create inf", h_ngram, "0.0"
    text=file.readline()
    if (not text) or ((len(text.split())==1) and (("-grams:" in text) or (text[:5] == "\\end\\"))): break
  ngrams.append(this_ngrams)

while (text and text[:5] != "\\end\\"): text=file.readline()
if not text:
  print "invalid ARPA file"
  sys.exit()
file.close()
#print text,

#fourgram "maxent" model (b(ABCD)=0):
#p(A)+b(A) A 0
#p(AB)+b(AB)-b(A)-p(B) AB 0
#p(ABC)+b(ABC)-b(AB)-p(BC) ABC 0
#p(ABCD)+b(ABCD)-b(ABC)-p(BCD) ABCD 0

#fourgram reverse ARPA model (b(ABCD)=0):
#p(A)+b(A) A 0
#p(AB)+b(AB)-p(B)+p(A) BA 0
#p(ABC)+b(ABC)-p(BC)+p(AB)-p(B)+p(A) CBA 0
#p(ABCD)+b(ABCD)-p(BCD)+p(ABC)-p(BC)+p(AB)-p(B)+p(A) DCBA 0

# compute new reversed ARPA model
print "\\data\\"
for n in range(1,len(cngrams)+1): # unigrams, bigrams, trigrams
  print "ngram "+str(n)+"="+str(len(ngrams[n-1].keys()))
offset = 0.0
for n in range(1,len(cngrams)+1): # unigrams, bigrams, trigrams
  print "\\"+str(n)+"-grams:"
  keys = ngrams[n-1].keys()
  keys.sort()
  for ngram in keys:
    prob = ngrams[n-1][ngram]
    # reverse word order
    words = ngram.split()
    rstr = " ".join(reversed(words))
    # swap <s> and </s>
    rev_ngram = rstr.replace("<s>","<temp>").replace("</s>","<s>").replace("<temp>","</s>")

    revprob = prob[0]
    if (prob[1] != inf): # only backoff weights from not newly created ngrams
      revprob = revprob + prob[1]
    #print prob[0],prob[1]
    # sum all missing terms in decreasing ngram order
    for x in range(n-1,0,-1): 
      l_ngram = " ".join(words[:x]) # shortened ngram
      if l_ngram not in ngrams[x-1]:
        sys.stderr.write(rev_ngram+": not found "+l_ngram+"\n")
      p_l = ngrams[x-1][l_ngram][0]
      #print p_l,l_ngram
      revprob = revprob + p_l

      r_ngram = " ".join(words[1:1+x]) # shortened ngram with offset one
      if r_ngram not in ngrams[x-1]:
        sys.stderr.write(rev_ngram+": not found "+r_ngram+"\n")
      p_r = ngrams[x-1][r_ngram][0]
      #print -p_r,r_ngram
      revprob = revprob - p_r

    if n != len(cngrams): #not highest order
      back = 0.0
      if rev_ngram[:3] == "<s>": # special handling since arpa2fst ignores <s> weight
        if n == 1:
          offset = revprob # remember <s> weight
          revprob = sentprob # apply <s> weight from forward model
          back = offset
        elif n == 2:
          revprob = revprob + offset # add <s> weight to bigrams starting with <s>
      if (prob[1] != inf): # only backoff weights from not newly created ngrams
        print revprob,rev_ngram.encode("utf-8"),back
      else:
        print revprob,rev_ngram.encode("utf-8"),"-100000.0"
    else: # highest order - no backoff weights
      if (n==2) and (rev_ngram[:3] == "<s>"): revprob = revprob + offset
      print revprob,rev_ngram.encode("utf-8")
print "\\end\\"
