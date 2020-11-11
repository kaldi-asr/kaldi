#!/usr/bin/env bash
# Copyright    2018   Desh Raj
# Apache 2.0

## Download all written works of Jeremy Bentham for the Bentham HWR task LM training

baseurl='http://oll.libertyfund.org/titles/'
savedir=$1

mkdir -p $savedir

declare -a texts=("bentham-the-works-of-jeremy-bentham-vol-1/simple"
                "bentham-the-works-of-jeremy-bentham-vol-2/simple"
                "bentham-the-works-of-jeremy-bentham-vol-3/simple"
                "bentham-the-works-of-jeremy-bentham-vol-5-scotch-reform-real-property-codification-petitions/simple"
                "bentham-the-works-of-jeremy-bentham-vol-6/simple"
                "bentham-the-works-of-jeremy-bentham-vol-7-rationale-of-judicial-evidence-part-2/simple"
                "bentham-the-works-of-jeremy-bentham-vol-8/simple"
                "bentham-the-works-of-jeremy-bentham-vol-9-constitutional-code"
                "bentham-the-works-of-jeremy-bentham-vol-10-memoirs-part-i-and-correspondence/simple"
                "bentham-the-works-of-jeremy-bentham-vol-11-memoirs-of-bentham-part-ii-and-analytical-index")

counter=1
for i in "${texts[@]}"
do
    echo "Downloading $baseurl$i"
    curl -s -N {$baseurl}{$i} | sed -e 's/<[^>]*>//g' > $savedir"/bentham"$counter".txt"
    ((counter++))
done

cat $savedir"/*.txt" > $savedir"/complete.txt"
rm $savedir"/bentham*.txt"
