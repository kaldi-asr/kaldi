#!/bin/bash

cat $1 | perl -pe 's/ - / /g' | \
perl -pe 's/([a-zæøå0-9][a-zæøå0-9][a-zæøå0-9][a-zæøå0-9][\.\?\!]) ([A-ZÆØÅ])/\1\n\2/g' | \
perl -pe 's/_nl?_|_NL_/ _NL_ \n/g' | \
perl -pe 's/_ns_|_NS_/ _NS_ \n/g' | \
perl -pe 's/([a-zæøå0-9]\!|\?)/\1\n/g' 
