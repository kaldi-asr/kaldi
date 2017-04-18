


echo "Format openasr-data-01-raw -> openasr-data-01"
srcdir=/data/openasr-data-01-raw
todir=/data/openasr-data-01
mkdir -p $todir/{data/train,data/test,doc,tmp}

find $srcdir -iname "*raw" >$todir/tmp/raws.find
# find $srcdir -iname "transcriptions" >$todir/tmp/trans.find
cat $srcdir/*/transcriptions >$todir/tmp/trans.txt

wc -l $todir/tmp/*

chmod +x local/*

python local/format_openasr_corpus.py $todir/tmp/raws.find $todir/tmp/trans.txt \
	$todir/data/train $todir/doc || exit 1;

echo "Parallel to wav(long time)..."
parallel -j12 $todir/doc/ffmpeg.cmd || exit 1;

echo "===== TODO Split Train/Test ====="

# echo "$0 DONE!"
exit 0
