while read line;
do

  id=$(echo "$line" | sed 's/^.*\(([^)]\+)\)$/\1/')
  echo $(echo "$line" | sed 's/^\(.*\)([^)]\+)$/\1/' | sed 's/^ *//' | sed 's/ *$//' | tr ' ' '~' | sed 's/\(.\{1\}\)/ \1 /g' | sed 's/^ *//' | sed 's/ *$//')" $id"
done
