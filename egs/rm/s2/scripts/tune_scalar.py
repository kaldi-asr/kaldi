#!/usr/bin/python -u

import os, sys, re, os.path

import findmax


# arguments
if(not len(sys.argv) == 6):
    print 'Usage: tune_scalar.py <decode.sh> "<decode_script> --arg %g" <decode_dir> <min> <max>'
    print '  the tuned value is injected to argument "<decode_script> --arg %g"'
    print '  <decode_script> must accept argument --arg followed by number'
    sys.exit(1)

decode = sys.argv[1]
script = sys.argv[2]
dir = sys.argv[3]
min = float(sys.argv[4])
max = float(sys.argv[5])

#debug
print decode, script, dir, min, max
log='%s/tune_scalar.log'%dir
log2='%s/tune_scalar.log.last_decode'%dir
os.system('echo "-------------tuning-started" >%s'%(log))

# function that we want to maximize
def tryScalarValue(value):
    print 'Trying %3.6f ... ' % value ,
    #clean-up previous decode
    os.system('rm -r %s/decode* 2>/dev/null' % dir)
    os.system('rm %s/wer 2>/dev/null' % dir)
    #substitute scalar value to <decode_script> argument
    script_value = script % value
    #build the command
    cmd = '%s "%s" %s >%s' % (decode,script_value,dir,log2)
    os.system('echo "%s " >>%s'%(cmd,log))
    #run the script
    os.system(cmd)
    #read the wer file
    f = open(dir+'/wer','r')
    line = f.readline().rstrip()
    f.close()
    #parse ...
    print line,
    os.system('echo "%s" >>%s'%(line,log))
    arr = line.split(' ')
    try:
      acc = 100.0-float(arr[3])
      print 'acc:%f' % acc
      return acc
    except:
      print 'Error: Could not find WER value!!!'
      raise NoWerFoundError 


# run the maximization
value = findmax.findMax(tryScalarValue,min,max)
print 'Guessing %g' % value
tryScalarValue(value)
print 'Best-value=%g' % value
print '%g' % value

