#!/bin/bash

targetDir=$1

echo "------------------------------------"
echo "Building an SRILM in \"$targetDir\""
echo "------------------------------------"

for f in $targetDir/vocab $targetDir/text.train $targetDir/text.dev; do
  [ ! -f $f ] && echo "$0: requires $f" && exit 1;
done

echo "-------------------"
echo "Good-Turing 3grams"
echo "-------------------"
ngram-count -lm $targetDir/3gram.gt011.gz -gt1min 0 -gt2min 1 -gt3min 1 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.gt012.gz -gt1min 0 -gt2min 1 -gt3min 2 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.gt022.gz -gt1min 0 -gt2min 2 -gt3min 2 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.gt023.gz -gt1min 0 -gt2min 2 -gt3min 3 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort

echo "-------------------"
echo "Kneser-Ney $targetDir/3grams"
echo "-------------------"
ngram-count -lm $targetDir/3gram.kn011.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.kn012.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.kn022.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/3gram.kn023.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 3 -order 3 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort


echo "-------------------"
echo "Good-Turing 4grams"
echo "-------------------"
ngram-count -lm $targetDir/4gram.gt0111.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 1 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0112.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0122.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0123.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0113.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0222.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.gt0223.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort

echo "-------------------"
echo "Kneser-Ney 4grams"
echo "-------------------"
ngram-count -lm $targetDir/4gram.kn0111.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 1 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0112.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0113.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0122.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0123.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0222.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort
ngram-count -lm $targetDir/4gram.kn0223.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text $targetDir/text.train -vocab $targetDir/vocab -unk -sort

echo "-------------------"
echo "Computing perplexity"
echo "-------------------"
for f in $targetDir/3gram* ; do echo $f; ngram -order 3 -lm $f -unk -ppl $targetDir/text.dev; done | tee $targetDir/perplexities.3gram
for f in $targetDir/4gram* ; do echo $f; ngram -order 4 -lm $f -unk -ppl $targetDir/text.dev; done | tee $targetDir/perplexities.4gram

