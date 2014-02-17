#!/bin/bash

### FSTs ###

# Fst fair_bet
# Inspiration from http://vimeo.com/7303679
# Casino has two coins. 
# The first one is fair the second one is loaded.
# It means that from the fair one we get with equal 
# probability of head (H) and tail (T).
# With the loaded one we get head with 0.6 probability
# and the tail with 0.4 probability.
# The problem is to guess when the casino plays with the loaded coin
# when the coin is fair.
# The priors (the chance that casino starts with fair or loaded coin) are:
# Fair:0.66666667  Loaded(L): 0.33333333
# The transition probabilities between states F and L
#        Fair  Loaded 
# Fair   0.95  0.05
# Loaded 0.9   0.1

# We saw a sequence H,H,T,T,H,T,H,H,H,T,H,H,T,T,H,T of tosses. 
# The task is to guess one of the state F or L for each toss. 

# fair_bet graph represents TODO
# We know:
# P(T|L) = 1- P(H|L) = 0.4   => P(H|L) = 0.6
# P(T|F) = P(H|F) = 0.5

# P(F_F) = 0.95
# P(L_L) = 0.9
cat > fair_bet.txt << FST
0 1 0 1 0.66666667
0 2 0 2 0.33333333
1 3 
FST


cat > non_symetric.txt << FST
0 1 0 1 10.0
0 2 0 2 10.0
1 3 1 3 15.0
1 4 1 4 5.0
2 3 2 3 20.0
2 4 2 4 50.0
3 5 3 5 20.0
4 5 4 5 10.0
5 0.0 
FST

cat > symetric.txt << FST
0 1 0 1 10.0
0 2 0 2 10.0
1 3 1 3 10.0
1 4 1 4 10.0
2 3 2 3 10.0
2 4 2 4 10.0
3 5 3 5 10.0
4 5 4 5 10.0
5 0.0 
FST

cat > symetric_end.txt << FST
0 1 0 1 10.0
0 2 0 2 10.0
1 3 1 3 10.0
1 4 1 4 10.0
2 3 2 3 10.0
2 4 2 4 10.0
3 5 3 5 10.0
4 5 4 5 10.0
5 11.0 
FST

cat > symetric_middle.txt << FST
0 1 0 1 10.0
0 2 0 2 10.0
1 3 1 3 10.0
1 4 1 4 10.0
2 3 2 3 10.0
2 4 2 4 10.0
3 11.0
4 11.0
3 5 3 5 10.0
4 5 4 5 10.0
FST

cat > negative.txt << FST
0 1 0 1 0.5
0 1 0 1 1.5
1 2 1 2 2.5
1 -0.5
2 3.5
FST

cat > negative_end.txt << FST
0 1 0 1 0.5
0 1 0 1 1.5
1 2 1 2 2.5
1 3.5 
2 -2.0
FST


### script ###
fsts="non_symetric symetric symetric_end symetric_middle negative negative_end fair_bet"

for name in $fsts; do
    fstcompile --arc_type=log ${name}.txt ${name}.fst
    fstdraw --portrait=true ${name}.fst | \
        dot -Tsvg  > ${name}.svg
done

exit 0
