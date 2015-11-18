#!/usr/bin/env python
import sys, os, logging, numpy as np
import shutil
import parse_config
from optparse import OptionParser

def parse_config_dict(config_file, d_args=[]):
  lines = map(lambda x: x.strip(), open(config_file, "r").readlines())
  for line in lines:
    line = line.lstrip() 
    if line[0:2] == "--":
      line = line.split("#")[0]
      line = line.strip()
      d_args.append(line)
  return d_args


def nnet_donefile_options(parser):

  parser.add_option('--train-error', dest="train_error", help="[default: %default]", default=np.inf, type=float)
  parser.add_option('--train-accuracy', dest="train_accuracy", help="[default: %default]", default=0.0, type=float)
  parser.add_option('--cv-error', dest="cv_error", help="[default: %default]", default=np.inf, type=float)
  parser.add_option('--cv-accuracy', dest="cv_accuracy", help="[default: %default]", default=0.0, type=float)
  parser.add_option('--learn-rate', dest="learn_rate", help="[default: %default]", default=0.008, type=float)
  parser.add_option('--nSamples', dest="nSamples", help="[default: %default]", default=0, type=int)

  return parser

## MAIN ##

usage = "%prog [options] out_done_file inp_done_files"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')


(o, args) = parser.parse_args()

if len(args) < 3:
  parser.print_help()
  sys.exit(1)

out_done_file = args[0]
inp_done_files_list = args[1:]

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)
logging.info(" combining done files")

inp_po = nnet_donefile_options(OptionParser())

out_nSamples = 0
(out_cv_error, out_cv_accuracy) = (0.0, 0.0)
(out_train_error, out_train_accuracy) = (0.0, 0.0)

#TODO: make it binary, make it soft
for ii, inp_done_file in enumerate(inp_done_files_list):
  logging.info("   from file %s", inp_done_file)
  (inp_stats, a) = inp_po.parse_args(parse_config_dict(inp_done_file))

  #nSamples
  out_nSamples += inp_stats.nSamples

  #cv_error
  #if inp_stats.cv_error != np.inf:
  out_cv_error += inp_stats.cv_error

  #train_error
  #if inp_stats.train_error != np.inf:
  out_train_error += inp_stats.train_error

  #cv_accuracy
  out_cv_accuracy += inp_stats.cv_accuracy

  #train_accuracy
  out_train_accuracy += inp_stats.train_accuracy

  #logging.info("%d, %f, %f, %f, %f", out_nSamples, out_cv_error, out_train_error, out_cv_accuracy, out_train_accuracy)

done_fd = open(out_done_file, "w")
if out_nSamples == 0:
  logging.error(" out_nSamples = 0")
  sys.exit(1)
done_fd.write("--nSamples=%d\n" %(out_nSamples))

if out_cv_error != np.inf:
  done_fd.write("--cv-error=%f\n" %( out_cv_error))

if out_train_error != np.inf:
  done_fd.write("--train-error=%f\n" %( out_train_error))

if out_cv_accuracy != 0:
  done_fd.write("--cv-accuracy=%f\n" %( out_cv_accuracy))

if out_train_accuracy != 0:
  done_fd.write("--train-accuracy=%f\n" %( out_train_accuracy))

done_fd.close()

import time
time.sleep(60)

sys.exit(0)


