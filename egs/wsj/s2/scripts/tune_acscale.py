#!/usr/bin/python -u

import os, sys, re, os.path

import findmax


# arguments
if(not len(sys.argv) == 7):
    print 'SYNTAX: tunescale_acc.py min max work_directory decode_script graph scp'
    sys.exit(1)

min = float(sys.argv[1])
max = float(sys.argv[2])
dir = sys.argv[3]
script = sys.argv[4]
graph = sys.argv[5]
scp = sys.argv[6]

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
    os.system('cat %s | sed -e \'s|^\(acwt\)=.*|\\1=%f|\' > %s' %(script,scale,script_tgt))
    os.system('chmod u+x %s' %script_tgt)
    #run the script
    cmd= 'bash scripts/decode.sh %s_tmp/ %s %s %s &> /dev/null' % (dir,graph,script_tgt,scp)
    #print cmd
    os.system(cmd)
    #read the wer file
    f = open(dir+'_tmp/wer','r')
    line = f.readline().rstrip()
    line = f.readline().rstrip()
    f.close()
    #parse ...
    print line,
    arr = line.split(' ')
    try:
      acc = 100.0-float(arr[1])
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

