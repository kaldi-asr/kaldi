#!/usr/bin/python -u

import os, sys, re, os.path

import findmax


# arguments
if(not len(sys.argv) == 5):
    print 'SYNTAX: tunescale_acc.py min max work_directory decode_script'
    sys.exit(1)

min = float(sys.argv[1])
max = float(sys.argv[2])
dir = sys.argv[3]
script = sys.argv[4]

#modified original bash script
script_tgt = dir+'/'+os.path.basename(script)
#create dir when necessary
if(not os.path.exists(dir)):
    os.mkdir(dir)

#debug
print min, max, dir, script, script_tgt

# function that we want to maximize
def tryScaleValue(scale):
    print 'Trying %3.6f ... ' % scale ,

    #clean-up previous
    os.system('rm -f %s/*' % dir)
    #generate new script
    os.system('cat %s | sed -e \'s|^\(acousticscale\)=.*|\\1=%f|\' -e \'s|^\(dir\)=.*|\\1=%s|\' > %s' %(script,scale,dir,script_tgt))
    #run the script
    os.system('bash '+script_tgt+' &> /dev/null')
    #read the wer file
    f = open(dir+'/wer','r')
    line = f.readline().rstrip()
    f.close()
    #parse ...
    print line,
    arr = line.split(' ')
    try:
      acc = 100.0-float(arr[3])
      print 'acc:%f' % acc
      return acc
    except:
      print 'Error: Could not find WER value!!!'
      raise NoWerFoundError 


# run the maximization
scale = findmax.findMax(tryScaleValue,min,max)
print 'Guessing %g' % scale
tryScaleValue(scale)
print 'Scale=%g' % scale
print '%g' % scale

