#!/bin/bash

# Copyright 2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a data directory, such as in data/train/, and modifies
# the wav.scp to perturb the volume (typically useful for training data when
# using systems that don't have cepstral mean normalization).

reco2vol=
force=false
scale_low=0.125
scale_high=2

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <datadir>"
  echo "e.g.:"
  echo " $0 data/train"
  exit 1
fi

export LC_ALL=C

data=$1

if [ ! -f $data/wav.scp ]; then
  echo "$0: Expected $data/wav.scp to exist"
  exit 1
fi

volume_perturb_done=`head -n100 $data/wav.scp | python -c "
import sys, re
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  # Handle three cases of rxfilenames appropriately; 'input piped command', 'file offset' and 'filename'
  parts = line.strip().split()
  if line.strip()[-1] == '|':
    if re.search('sox --vol', ' '.join(parts[-11:])):
      print 'true'
      sys.exit(0)
  elif re.search(':[0-9]+$', line.strip()) is not None:
    continue
  else:
    if ' '.join(parts[1:3]) == 'sox --vol':
      print 'true'
      sys.exit(0)
print 'false'
"` || exit 1

if $volume_perturb_done; then
  echo "$0: It looks like the data was already volume perturbed.  Not doing anything."
  exit 0
fi

if [ -z "$reco2vol" ]; then
  cat $data/wav.scp | python -c "
import sys, os, subprocess, re, random
random.seed(0)
scale_low = $scale_low
scale_high = $scale_high
volume_writer = open('$data/reco2vol', 'w')
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  # Handle three cases of rxfilenames appropriately; 'input piped command', 'file offset' and 'filename'
  vol = random.uniform(scale_low, scale_high)

  parts = line.strip().split()
  if line.strip()[-1] == '|':
    print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), vol)
  elif re.search(':[0-9]+$', line.strip()) is not None:
    print '{id} wav-copy {wav} - | sox --vol {vol} -t wav - -t wav - |'.format(id = parts[0], wav=' '.join(parts[1:]), vol = vol)
  else:
    print '{id} sox --vol {vol} -t wav {wav} -t wav - |'.format(id = parts[0], wav=' '.join(parts[1:]), vol = vol)
  volume_writer.write('{id} {vol}\n'.format(id = parts[0], vol = vol))
"  > $data/wav.scp_scaled || exit 1;
else
  cat $data/wav.scp | python -c "
import sys, os, subprocess, re
volumes = {}
for line in open('$reco2vol'):
  if len(line.strip()) == 0:
    continue
  parts = line.strip().split()
  volumes[parts[0]] = float(parts[1])

for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  # Handle three cases of rxfilenames appropriately; 'input piped command', 'file offset' and 'filename'
  
  parts = line.strip().split()
  id = parts[0]

  if id not in volumes:
    raise Exception('Could not find volume for id {id}'.format(id = id))

  vol = volumes[id]

  if line.strip()[-1] == '|':
    print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), vol)
  elif re.search(':[0-9]+$', line.strip()) is not None:
    print '{id} wav-copy {wav} - | sox --vol {vol} -t wav - -t wav - |'.format(id = parts[0], wav=' '.join(parts[1:]), vol = vol)
  else:
    print '{id} sox --vol {vol} -t wav {wav} -t wav - |'.format(id = parts[0], wav=' '.join(parts[1:]), vol = vol)
"  > $data/wav.scp_scaled || exit 1;

  cp $reco2vol $data/reco2vol
fi

len1=$(cat $data/wav.scp | wc -l)
len2=$(cat $data/wav.scp_scaled | wc -l)
if [ "$len1" != "$len2" ]; then
  echo "$0: error detected: number of lines changed $len1 vs $len2";
  exit 1
fi

mv $data/wav.scp_scaled $data/wav.scp

if [ -f $data/feats.scp ]; then
  echo "$0: $data/feats.scp exists; moving it to $data/.backup/ as it wouldn't be valid any more."
  mkdir -p $data/.backup/
  mv $data/feats.scp $data/.backup/
fi

echo "$0: added volume perturbation to the data in $data"
exit 0

