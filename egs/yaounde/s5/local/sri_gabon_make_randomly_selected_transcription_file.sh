#!/bin/bash
#  concatenate 4 copies of the file
cat local/src/sri_gabon_potential_prompts.txt \
    local/src/sri_gabon_potential_prompts.txt \
    local/src/sri_gabon_potential_prompts.txt \
    local/src/sri_gabon_potential_prompts.txt > \
    data/local/tmp/sri_gabon_potential_prompts.txt

# shuffle the potential prompts
shuf data/local/tmp/sri_gabon_potential_prompts.txt > \
     data/local/tmp/sri_gabon_potential_prompts_shuffled.txt

# skim off the top 7417lines
head -n 7417 \
     data/local/tmp/sri_gabon_potential_prompts_shuffled.txt > \
     data/local/tmp/sri_gabon_randomly_selected_prompts.txt

# number the selected lines
nl \
    data/local/tmp/sri_gabon_randomly_selected_prompts.txt > \
    data/local/tmp/sri_gabon_randomly_selected_prompts_nl.txt

# fix the numbering
{
    while read line; do
	num=$(echo "$line" | cut -f 1 | tr -d " ")
	prompt=$(echo "$line" | cut -f 2)
	if [ $num -le 9 ]; then
	    echo "000$num	$prompt";
	elif [ $num -le 99 ]; then
	    echo "00$num	$prompt"
	elif [ $num -le 999 ]; then
	    echo "0$num	$prompt"
	else
	    echo "$num	$prompt"
	fi
    done
} < data/local/tmp/sri_gabon_randomly_selected_prompts_nl.txt > \
  data/local/tmp/sri_gabon_prompts.txt
