###############################
#  Examples commands for gdb  #
###############################
# layout src  # Open TUI mode in viewing src code
# file ./online-python-gmm-decode-faster-test # we want to debug this program
# directory ../feat # add directory to search path for source codes
# b online-python-gmm-decode-faster-test.cc:80 # set breakpoint
# set args --rt-min=0.5 --rt-max=0.7   # pass arguments to program
# shell ls  # execute shell command ls

#############################
#  EXAMPLES how to run gdb  #
#############################
# gdb -q -iex "set auto-load safe-path ." .gdbinit
# gdb -q -x .gdbinit

#######################
#  USEFULL shortcuts  #
#######################
# Ctrl+x Ctrl+a ... switches of and of tui from gdb prompt
# In TUI mode: Ctrl+n  resp. Ctrl+p ... next resp. previous line in history

###########
#  LINKS  #
###########
# Martin Jiricka slides in czech:
# http://www.ms.mff.cuni.cz/~jirim7am/data/gdb/gdb.pdf (or in my Calibre library)

directory ../../../dec-wrap
directory ../../../decoder
directory ../../../feat

# setup
# b pykaldi-faster-wrapper.cc:175
# b pykaldi-faster-wrapper.cc:220

# decode
# b pykaldi-faster-wrapper.cc:74

# decode - decoder
# b pykaldi-faster-decoder.cc:46

# extractor->Compute
b pykaldi-feat-input.h:166
# display frame
b feature-mfcc.cc:75

run
