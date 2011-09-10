#!/bin/awk -f

{
gsub("\\[NOISE\\]","<NOISE>",$0);
gsub("\\[SILENCE\\]","<NOISE>",$0);

gsub("\\[VOCALIZED-NOISE\\]","<SPOKEN_NOISE>",$0);
gsub("\\[LAUGHTER\\]","<SPOKEN_NOISE>",$0);

#gsub("\\[laughter.*\\]","<SPOKEN_NOISE>",$0);
#gsub("[^ ]*\\[.*\\][^ ]*","<SPOKEN_NOISE>",$0);

print;
}
