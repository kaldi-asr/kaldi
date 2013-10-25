#! /usr/bin/bash

VARIABLES=`diff <(compgen -A variable) <(. ./lang.conf.orig; compgen -A variable) | grep '^>'| sed 's/^> *//g'`
. ./lang.conf.orig

for variable in $VARIABLES ; do

	eval VAL=\$${variable}
	if [[ $VAL =~ /export/babel/data/ ]] ; then
		eval export $variable=${VAL/${BASH_REMATCH[0]}/"/work/02359/jtrmal/"/}
		declare -x $variable
	fi
done
train_nj=24
dev10h_nj=60

