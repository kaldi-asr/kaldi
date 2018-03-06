for it in 1 2 3 4 5 6 7 8
do
  /export/b01/babak/kaldi/kaldi/src/bin/ali-to-phones --write-lengths exp/tri2b_pregdl_ali/final.mdl "ark:gunzip -c exp/tri2b_pregdl_ali/ali.$it.gz|" ark,t:exp/gdl/al/$it.tra
done

cat exp/gdl/al/*.tra | cut -d' ' -f2- | tr ';' "\n" | sed 's/^ *//' | sed 's/ *$//' | egrep -v '^$' | sort -g -k1 > tmp
maxId=$(tail -n 1 tmp | cut -d' ' -f1)
for id in $(seq 1 $maxId)
do
  echo $id" "$(cat tmp | egrep "^$id " | cut -d' ' -f2- |tr "\n" ' ' | sed 's/^ *//' | sed 's/ *$//' | gawk '{tmp=$0;gsub(/ /,"+",tmp);print("scale=10;("tmp")/"NF)}'|bc )
done
