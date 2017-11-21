#!/usr/bin/perl -p

# handles nonprinting characters in Broadcast News material, to the extent
# that they can be handled, and perhaps a bit beyond...

s=\xc4=-=g;
s=\xae=<<=g;
s=\xaf=>>=g;
s=\x82=e=g;	# e' (é) in IBMPC
s=\xab= 1/2=g;
# next most frequent, \xfa, appears to have various use as hard-space,
#  hard-return, or noise
s=\x90=E=g;	# E' (É) in IBMPC
s=\xa4=n=g;	# n~ (ñ) in IBMPC
s=\xac= 1/4=g;
# ^G => noise
# ^A => noise
s=\xf8= degrees=g;
# \x1b => noise?
# \x02 => noise?

# remainder occur 4 or fewer times each -- may be better to do by hand?
s=\x89=e=g;	# e: or E:
s=\xf1= plus or minus =g;
# \xc9 = graphics character => ???
# \x03 => noise?
# \x04 => noise?
s=\x8a=e=g;	# e` (è) in IBMPC
s=\x87=c=g;	# c, (ç) in IBMPC
s=\xe9=e=g;	# e' (é) in ISO!!
# \xad => spanish inverted question mark (¡), appears (with Spanish) twice!
s=\xad==g;

# remainder occur only once each -- probably best to check by hand
# \xff
# \xdd
# \xbb
# \xa1
# \x8d
# \x81
# \x1c
# \x1a
# \x16
# \x11
# \x10
# \x0c
