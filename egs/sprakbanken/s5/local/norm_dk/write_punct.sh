#!/bin/bash

perl -pe 's/([\n ])\.([ \n])/\1PUNKTUM\2/g' | \
perl -pe 's/([\n ])\:([ \n])/\1KOLON\2/g' | \
perl -pe 's/([\n ])\;([ \n])/\1SEMIKOLON\2/g' | \
perl -pe 's/([\n ])_NL_([ \n])/\1NY LINJE\2/g' | \
perl -pe 's/([\n ])_NS_([ \n])/\1NYT AFSNIT\2/g' | \

tr -s ' '