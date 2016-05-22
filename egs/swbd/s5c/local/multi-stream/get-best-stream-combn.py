#!/usr/bin/env python
import sys, os, logging, numpy as np
import numpy.matlib
import cPickle as pickle, bz2
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils/numpy_io'))
import kaldi_io

import time

def keep_topN(score_list, topN=1):

  if topN == -1:
    return score_list
  
  #topN_ind = np.argpartition(score_list, -topN)[-topN:]
  topN_ind = np.argsort(score_list)[::-1][:topN]
  
  for ind in xrange(0, len(score_list)):
    if ind not in topN_ind:
      score_list[ind] = 0

  return score_list

def get_utt_mat_from_scp(scp, utt):
  
  fd = kaldi_io.open_or_fd(scp)
  for line in fd:
    (key, rxfile) = line.split(' ')
    if key == utt:
      mat = kaldi_io.read_mat(rxfile)
      return mat
    

  print "unable to find "+utt
  print "exiting..... "
  
  sys.exit(1)

def binary_to_dec(bin_arr, base=2):

  # converts binary_arr to int
  dec = 0
  for ii, g in enumerate(bin_arr[::-1]):
    dec = dec + g*np.power(base, ii)
  return dec


def gen_child_from_root(root_combn, num_streams):
  
  root_bin_str = '{:040b}'.format(root_combn)
  root_bin_str = root_bin_str[-num_streams:]
  root_bin = np.asarray(map(lambda x: int(x), root_bin_str ))

  # gen child, i.e.
  # switch-off one 1 at a time 
  child_bin_list = []
  for ii, one_indx in enumerate(np.nonzero(root_bin)[0]):
    this_child_bin = np.copy(root_bin)
    this_child_bin[one_indx] = 0 #make it zero
    child_bin_list.append( this_child_bin)

  child_combn_list=[]  
  for this_child_bin in child_bin_list:
    this_child_combn = binary_to_dec(this_child_bin)
    child_combn_list.append(this_child_combn)

  return child_combn_list

def get_score_from_combn(utt, combn):
  
  combn_score_file = aann_scores_dir+"/comb."+str(combn)+".pklz"

  if os.path.exists(combn_score_file):
    logging.info("%s exists, loading ... ", combn_score_file)
    # load combn file
    d = pickle.load(bz2.BZ2File(combn_score_file, "rb"))
  else:
    logging.info("%s NOT exists, computing scores for this combn ... ", combn_score_file)
    # run score generation script
    logfile=logdir+"/compute_stream_combn_scores."+str(combn)+".log"
    logfile_f=open(logfile, 'w')

    cmd_str="bash "+compute_stream_combn_scores_task+" "+str(combn)
    ret_status = subprocess.call(cmd_str, shell=True, stdout=logfile_f, stderr=logfile_f)

    if ret_status != 0:
      logging.info(" Error in computing score for utt=%s using combn=%d", utt, combn)
      sys.exit(1)

    d = pickle.load(bz2.BZ2File(combn_score_file, "rb"))

  try:
    return d[utt]
  except KeyError:
    logging.info(" Cant find score for utt=%s , in combn_score_file=%s", utt, combn_score_file)
    sys.exit(1)

      
def main_func(utt, parent_combn):

  # get parent_score
  parent_score = get_score_from_combn(utt, parent_combn)

  # get children
  child_combn_list = gen_child_from_root(parent_combn, num_streams)
  child_combn_list = np.asarray(child_combn_list)

  # break conditions
  if len(child_combn_list) == 0:
    return (parent_combn, parent_score)
  if len(child_combn_list) == 1:
    child_combn = binary_to_dec(child_combn_list[0])
    if child_combn == 0:
      return (parent_combn, parent_score)
    
  child_combn_score_list = []
  for this_child_combn in child_combn_list:
    this_child_combn_score = get_score_from_combn(utt, this_child_combn)
    child_combn_score_list.append(this_child_combn_score)
  child_combn_score_list = np.asarray(child_combn_score_list)

  # modify child_combn_score_list, so that keep only ones
  # satisfying this_child_combn_score > alpha * parent_score 
  child_combn_score_list[ child_combn_score_list <= alpha * parent_score ] = 0.0

  # modify child_combn_score_list retain only topN, make rest=zero
  ## print >> sys.stderr, child_combn_score_list
  child_combn_score_list = keep_topN(child_combn_score_list, topN)
  print >> sys.stderr, "  parent_combn="+str(parent_combn)+",  parent score="+str(parent_score)
  print >> sys.stderr, "  Keeping combn="
  print >> sys.stderr, child_combn_list[child_combn_score_list != 0]
  print >> sys.stderr, child_combn_list
  print >> sys.stderr, child_combn_score_list

  (subtree_combn_list, subtree_combn_score_list) = ([], [])
  for ii, this_child_combn in enumerate(child_combn_list):
    this_child_combn_score = child_combn_score_list[ii] #get_score_from_combn(utt, this_child_combn)
    # print this_child_combn, this_child_combn_score

    if this_child_combn_score > parent_score:
      # time.sleep(10)
      (t1, t2) = main_func(utt, this_child_combn)

      subtree_combn_list.append(t1)
      subtree_combn_score_list.append(t2)

  # return best of child's scores
  (best_subtree_combn, best_subtree_combn_score) = (parent_combn, parent_score)
  for ii, subtree_combn_score in enumerate(subtree_combn_score_list):
    if subtree_combn_score > best_subtree_combn_score:
      best_subtree_combn = subtree_combn_list[ii]
      best_subtree_combn_score = subtree_combn_score

  return (best_subtree_combn, best_subtree_combn_score)

from optparse import OptionParser
usage = "%prog [options] <aann_scores_dir> <num_streams> <feat-to-len-list> <compute_stream_combn_scores.task> <logdir>"
parser = OptionParser(usage)

parser.add_option('--alpha', dest='alpha',
                  help='score_child > alpha * score_parent [default: %default]',
                  default=1, type='float');

parser.add_option('--topN', dest='topN',
                  help='traverse through topN of child satisfying score_child > alpha * score_parent [default: %default]',
                  default=-1, type='int');

(o, args) = parser.parse_args()
if len(args) != 5:
  parser.print_help()
  sys.exit(1)

## Create log file
logging.basicConfig(stream=sys.stderr, format='%(asctime)s: %(message)s', level=logging.INFO)

logging.info(" Running as %s ", sys.argv[0])
logging.info(" %s", " ".join(sys.argv))

(aann_scores_dir, num_streams, feat_to_len_list, compute_stream_combn_scores_task, logdir)=(args[0], int(args[1]), args[2], args[3], args[4])

# read eval scores
num_combns=np.power(2, num_streams)-1
all_stream_combn=num_combns

alpha = float(o.alpha)
topN = int(o.topN)

out_ark='/dev/stdout'
with open(out_ark, 'wb') as output:
  for line in open(feat_to_len_list, "r").readlines():
    line = line.rstrip()
    (utt, num_frames)  = (line.split()[0], int(line.split()[1]))
  
    (best_combn, best_combn_score) = main_func(utt, all_stream_combn) 

    #best_comb -> strm_mask    
    bin_str='{:040b}'.format(best_combn)
    bin_str=bin_str[-num_streams:]
    wts = np.asarray(map(lambda x: int(x), bin_str))
    
    logging.info("utt=%s , best_comb=%d, %s", utt, best_combn, wts)
    
    Wts = np.matlib.repmat(wts, num_frames, 1)

    kaldi_io.write_mat(output, Wts, key=utt)

sys.exit(0)


