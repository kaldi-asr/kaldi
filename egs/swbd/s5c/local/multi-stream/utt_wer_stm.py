#!/usr/bin/env python

import sys, os, logging

# Append Levenshtein alignment of 'hypothesis' and 'reference' into 'CTM':
# (parsed from the 'prf' output of 'sclite')

# The tags in appended column are:
#  'C' = correct
#  'S' = substitution
#  'I' = insertion
#  'U' = unknown (not part of scored segment)

def compute_interval_overlap((t1, t2), (t1_hat, t2_hat)):
  
  if not (t1 < t2):
    print t1, t2
  assert t1 < t2
  
  if not (t1_hat < t2_hat):
    print t1_hat, t2_hat
  assert t1_hat < t2_hat

  if t2 < t1_hat or t1 > t2_hat: #no overlap case
    return 0

  if t1 <= t1_hat and t2 <= t2_hat:
    return t2 - t1_hat

  if t1 >= t1_hat and t2 >= t2_hat:
    return t2_hat - t1

  if t1 >= t1_hat and t2 <= t2_hat:
    return t2 - t1

  if t1 <= t1_hat and t2 >= t2_hat:
    return t2_hat - t1_hat


###############################################
def load_prf_file(prf_file):
  # Load the prf file,
  prf_dict = dict()
  lines = open(prf_file).readlines()
  last_stm_seg_id = ""
  for i in range(len(lines)):
    l = lines[i]

    if l[:3] == 'id:':
      stm_seg_id = l.split()[1].split('(')[1].split(')')[0]
      prf_dict[stm_seg_id] = dict()
      last_stm_seg_id = stm_seg_id

    if l[:5] == "File:":
      file_id = l.split()[1]
      prf_dict[last_stm_seg_id]['File:'] = file_id

    if l[:8] == 'Channel:':
      chan = l.split()[1]
      prf_dict[last_stm_seg_id]['Channel:'] = chan

    if l[:7] == 'Scores:':
      scores = map(lambda x: int(x), l.split()[5:])
      prf_dict[last_stm_seg_id]['Scores:'] = scores

    if l[:10] == 'Ref times:':
      (t1, t2) = (float(l.split()[3]), float(l.split()[5]))
      prf_dict[last_stm_seg_id]['Ref times:'] = (t1, t2)

    if l[:4] == 'REF:':
      prf_dict[last_stm_seg_id]['REF:'] = l.split()[1:]

    if l[:4] == 'HYP:':
      prf_dict[last_stm_seg_id]['HYP:'] = l.split()[1:]

    if l[:5] == 'H_T1':
      prf_dict[last_stm_seg_id]['H_T1:'] = l.split()[1:]

    if l[:5] == 'H_T2':
      prf_dict[last_stm_seg_id]['H_T2:'] = l.split()[1:]

    if l[:5] == 'CONF:':
      prf_dict[last_stm_seg_id]['CONF:'] = map( lambda x: float(x), l.split()[1:])

    if l[:5] == 'Eval:':
      prf_dict[last_stm_seg_id]['Eval:'] = l.split()[1:]

  return prf_dict

##########################
#### Read kaldi lines ####
##########################
def read_kaldi_segs(kaldi_segments):

  kaldi_segs_dict = dict()
  f = open(kaldi_segments)
  for l in f:
    (utt, reco_id, t1, t2) = (l.split()[0], l.split()[1], float(l.split()[2]), float(l.split()[3]))

    kaldi_segs_dict[utt] = dict()
    kaldi_segs_dict[utt]['File:'] = reco_id
    kaldi_segs_dict[utt]['Ref times:'] = (t1, t2)

  return kaldi_segs_dict


##########################
#### Read wav.map ########
##########################
def read_wav_map(wav_map):

  wav_map_dict = {}
  wav_map_revdict = {}
  if o.wav_map != "":
    f = open(o.wav_map)
    for l in f:
      (w1, w2) = (l.split()[0], l.split()[1].lower())
      wav_map_dict[w1] = w2
      wav_map_revdict[w2] = w1

  return (wav_map_dict, wav_map_revdict)


def get_segs_map(kaldi_segs_dict, prf_dict, wav_map_dict=dict()):

  segs_map = dict()

  for kaldi_seg_id in sorted(kaldi_segs_dict.keys()):
    match_list = ([], []) #list of stm_segs, amount_of_overlap
    for stm_seg_id in prf_dict.keys():
      if len(wav_map_dict.keys()) == 0:
        reco_id = kaldi_segs_dict[kaldi_seg_id]['File:']
      else:
        reco_id = wav_map_dict[kaldi_segs_dict[kaldi_seg_id]['File:']]

      #if reco_id.lower() == prf_dict[stm_seg_id]['File:'].lower():
      reco_id="_".join(reco_id.split('-'))
      if reco_id.lower() in stm_seg_id:
        (t1, t2) = kaldi_segs_dict[kaldi_seg_id]['Ref times:']
        (t1_hat, t2_hat) = prf_dict[stm_seg_id]['Ref times:']

        if t1_hat < t2_hat:
          amt_of_overlap = compute_interval_overlap((t1, t2), (t1_hat, t2_hat))

          match_list[0].append(stm_seg_id)
          match_list[1].append(amt_of_overlap)

    max_idx = match_list[1].index(max(match_list[1]))
    # print kaldi_seg_id, kaldi_segs_dict[kaldi_seg_id]['Ref times:'], match_list[0][max_idx], prf_dict[match_list[0][max_idx]]['Ref times:']
    segs_map[kaldi_seg_id] = (match_list[0][max_idx], prf_dict[match_list[0][max_idx]]['Ref times:'])
    
  return segs_map

def align_stm_decode_segments(kaldi_segments, prf_file, wav_map):

  prf_dict = load_prf_file(prf_file)
  kaldi_segs_dict = read_kaldi_segs(kaldi_segments)

  (wav_map_dict, wav_map_revdict) = (dict(), dict())
  if wav_map != "":
    (wav_map_dict, wav_map_revdict) = read_wav_map(o.wav_map)
  
  segs_map = get_segs_map(kaldi_segs_dict, prf_dict, wav_map_dict)

  
  return (prf_dict, kaldi_segs_dict, wav_map_dict, segs_map)


if __name__ == "__main__":

  ###############################################
  from optparse import OptionParser
  usage = "%prog [options] kaldi_segments sclite_prf_file scores_pkl"
  parser = OptionParser(usage)

  parser.add_option('--wav-map', dest="wav_map",
                    help="Provide if differnet names [default: %default]",
                    default="", type='string')

  (o, args) = parser.parse_args()

  if len(args) != 3:
    parser.print_help()
    sys.exit(1)

  (kaldi_segments, prf_file, scores_pkl) = (args[0], args[1], args[2])


  (prf_dict, kaldi_segs_dict, wav_map_dict, segs_map) = align_stm_decode_segments(kaldi_segments, prf_file, o.wav_map)

  # Get C, S, D, I  
  utt_scores = dict()
  for kaldi_seg_id in sorted(kaldi_segs_dict.keys()):
    (stm_seg_id, ref_times) = segs_map[kaldi_seg_id]
    (C, S, D, I) = prf_dict[stm_seg_id]['Scores:'] 

    # compute WER
    wer = ((S + D + I)/float(S + D + C))*100
    acc = 100 - wer

    utt_scores[kaldi_seg_id] = acc
    
  #pickle dump
  import bz2, cPickle as pickle
  
  f = bz2.BZ2File(scores_pkl, "wb")
  pickle.dump(utt_scores, f)
  f.close()

  sys.exit(0)

