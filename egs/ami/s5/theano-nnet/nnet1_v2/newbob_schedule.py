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

usage = "%prog [options] <nnetdir>"
parser = OptionParser(usage)

parser.add_option('--config', dest="config",
                  help="Configuration file to read (this option may be repeated) [default: %default]",
                  default="", type='string')

parser.add_option('--iter', dest="iter",
                  help="Current iteration [default: %default]",
                  default="0", type='string')

parser = parse_config.theano_nnet_parse_opts(parser)

(o, args) = parser.parse_args()
# options specified in config overides command line
if o.config != "": (o, args) = parse_config.parse_config(parser, o.config)

if len(args) != 1:
  parser.print_help()
  sys.exit(1)

nnetdir = args[0]
o.iter=int(o.iter)

## Create log file
logging.basicConfig(stream=sys.stdout, format='%(asctime)s: %(message)s', level=logging.INFO)
logging.info(" Newbob scheduling")

po = nnet_donefile_options(OptionParser())

best_cv_done_file="%s/.best_cv" %(nnetdir)
(best_cv_stats, a) = po.parse_args(parse_config_dict(best_cv_done_file))

new_cv_done_file="%s/.done_cv_iter%02d" %(nnetdir, o.iter)
(new_cv_stats, a) = po.parse_args(parse_config_dict(new_cv_done_file))

last_cv_error = best_cv_stats.cv_error / float(best_cv_stats.nSamples)

loss_prev = last_cv_error
loss_new = new_cv_stats.cv_error / float(new_cv_stats.nSamples)

logging.info(" best_cv_error = %f", last_cv_error)
logging.info(" new_cv_error = %f", loss_new)

this_nnet_file = "%s/nnet/nnet_iter%02d" %(nnetdir, o.iter)
if loss_new < loss_prev:
  last_cv_error = loss_new
  #copy to .best_cv
  shutil.copyfile(new_cv_done_file, best_cv_done_file)
  #copy to .best_nnet
  best_nnet_file = this_nnet_file
  best_nnet_fd = open("%s/.best_nnet" %(nnetdir), "w")
  best_nnet_fd.write("%s" %(best_nnet_file))
  best_nnet_fd.close()
  
else:
  # mv 
  shutil.move(this_nnet_file, this_nnet_file+"_rejected")
  logging.info("nnet rejected %s_rejected" %(this_nnet_file))

# stopping criterion
rel_impr = (loss_prev - last_cv_error)/loss_prev
if os.path.exists(nnetdir+"/.halving") and rel_impr < o.end_halving_impr:
  logging.info(" finished, too small rel. improvement %f", rel_impr)
  open(nnetdir+"/.finished", 'a').close()
  sys.exit(0)

# start annealing when improvement is low
if rel_impr < o.start_halving_impr:
  open(nnetdir+"/.halving", 'a').close()
  logging.info(" halving learn_rate")

#new_learn-rate
new_learn_rate = o.learn_rate
if os.path.exists(nnetdir+"/.halving"):
  new_learn_rate = new_learn_rate * o.halving_factor
fd = open(nnetdir+"/.learn_rate", 'w')
fd.write("%f" %(new_learn_rate))

logging.info(" next_learn_rate = %f", new_learn_rate)



