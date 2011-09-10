#!/bin/awk -f

{
segment=$1;
split(segment,S,"[_-]");
side=S[2];
audioname=S[1];
startf=S[3];
endf=S[4];

print segment " " audioname "-" side " " startf/100 " " endf/100
}