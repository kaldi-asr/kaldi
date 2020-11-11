#!/usr/bin/env python

# this script is to be used to convert MGB challenge xmls to SCLITE stm
# format so it can be used as reference for evaluation.
#
# Copyright (C) 2016, Qatar Computing Research Institute, HBKU (author: Yifan Zhang)
#

__author__ = 'Yifan Zhang (yzhang@qf.org.qa)'

import os
import sys
import time
import codecs
from xml.dom import minidom

_unicode = u"\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_buckwalter = u"|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"

_forwardMap = {ord(a):b for a,b in zip(_unicode, _buckwalter)}
_backwardMap = {ord(b):a for a,b in zip(_unicode, _buckwalter)}

def toBuckWalter(s):
  return s.translate(_forwardMap)

def fromBuckWalter(s):
  return s.translate(_backwardMap)

class Element(object):
  def __init__(self, text, startTime, endTime=None):
    self.text = text
    self.startTime = startTime
    self.endTime = endTime

def loadXml(xmlFileName, opts):
  dom = minidom.parse(open(xmlFileName, 'r'))
  trans = dom.getElementsByTagName('transcript')[0]
  segments = trans.getElementsByTagName('segments')[0]
  elements = []
  for segment in segments.getElementsByTagName('segment'):
    sid = segment.attributes['id'].value.split('_utt_')[0].replace("_","-")
    startTime = float(segment.attributes['starttime'].value)
    endTime = float(segment.attributes['endtime'].value)

    tokens = [e.childNodes[0].data for e in segment.getElementsByTagName('element') if len(e.childNodes)]
    # skip any word starts with '#'
    tokens = filter(lambda i: not i.startswith('#'), tokens)
    # convert to buckwalter if required
    if opts.buck:
      tokens = map(toBuckWalter, tokens)

    text = ' '.join(tokens)

    elements.append(Element(text, startTime, endTime))
  return {'id': sid, 'turn': elements}

def stm(data):
  out = codecs.getwriter('utf-8')(sys.stdout)
  for e in data['turn']:
    out.write("{} 1 UNKNOWN {:.02f} {:.02f} ".format(data['id'], e.startTime, e.endTime))
    out.write(e.text)
    out.write("\n")

def ctm(data):
  """ generate ctm output for test
  """
  out = codecs.getwriter('utf-8')(sys.stdout)
  for e in data['turn']:
    tokens = e.text.split()
    duration = e.endTime - e.startTime
    interval = duration / len(tokens)
    startTime = e.startTime
    for token in tokens:
      out.write("{} 1 {:.02f} {:.02f} ".format(data['id'], startTime, interval))
      out.write(token)
      out.write("\n")

def main(args):
  data = loadXml(args.xmlFileName, args)
  if args.ctm:
    ctm(data)
  else:
    stm(data)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='convert Arabic MGB xml file to MGB xml')
  parser.add_argument("--id", dest="uid",
                      help="utterance id")
  parser.add_argument("--buck", dest="buck", default=False, action='store_true',
                      help="output buckwalter text")
  parser.add_argument("--ctm", dest="ctm", default=False, action='store_true',
                      help="output ctm file for testing")
  parser.add_argument("--skip-bad-segments", dest="skip_bs", default=False, action='store_true',
                      help="skip segments with ###, these are either overlapped speech or unintelligible speech")
  parser.add_argument(dest="xmlFileName", metavar="xml", type=str)
  args = parser.parse_args()

  main(args)
