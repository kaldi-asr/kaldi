import os
import sys

def unique(m):
    unique_list = []

    for i in m:
        if i not in unique_list:
            unique_list.append(i)
    
    return unique_list

# Load in the MANIFEST file, save off the audio recoding file names
file = sys.argv[1]
prefix = '\thttps://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/'
m = []

with open(file) as f:
    for line in f:
        #splits = line.split('/')
        #print(splits)
        if line.startswith(prefix):
            splits = line.split('/')
            #print(splits)
            m.append(splits[7])
m = unique(m)
print("Got the audio files from MANIFEST.TXT")
#print(m)

# Separate files and save off into train, dev, and eval partitions
N = len(m)

train = m[:round(N*.5)]
dev = m[round(N*.5)+1:round(N*.8)]
eval = m[round(N*.8)+1:]

print("Train set: "+str(train))
print("Dev set: "+str(dev))
print("Eval set: "+str(eval))

if os.path.exists('local/split_train.orig'):
    os.remove('local/split_train.orig')
if os.path.exists('local/split_dev.orig'):
    os.remove('local/split_dev.orig')
if os.path.exists('local/split_eval.orig'):
    os.remove('local/split_eval.orig')

with open('local/split_train.orig', 'a') as train_file:
    for d in train:
        train_file.write(d)
        train_file.write("\n")

with open('local/split_dev.orig', 'a') as dev_file:
    for d in dev:
        dev_file.write(d)
        dev_file.write("\n")

with open('local/split_eval.orig', 'a') as eval_file:
    for d in eval:
        eval_file.write(d)
        eval_file.write("\n")
