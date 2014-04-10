#!/bin/bash

cat $1 | perl -pe 's/ - / /g' | \
perl -pe 's/_nl?_|_NL_/ _NL_ /g' | \
perl -pe 's/_ns_|_NS_/ _NS_ /g' | \
perl -pe 's/([a-zæøå0-9])(\!|\?)/\1 \2/g' 
